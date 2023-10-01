"""
File: batched_agent_manager.py
Author: Matthew Allen

Description:
    A class to manage the multi-processed agents interacting with instances of the environment. This class is responsible
    for spawning and closing the individual processes, interacting with them through their respective pipes, and organizing
    the trajectories from each instance of the environment.
"""

import multiprocessing as mp
import pickle
import selectors
import socket
import time
from typing import Union

import numpy as np
from numpy import frombuffer, prod
import torch

from rlgym_ppo.batched_agents import BatchedTrajectory, comm_consts
from rlgym_ppo.batched_agents.batched_agent import batched_agent_process
from rlgym_ppo.util import WelfordRunningStat

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterator, *args, **kwargs):
        return iterator


PACKET_MAX_SIZE = 8192


class BatchedAgentManager(object):
    def __init__(
        self,
        policy,
        min_inference_size=8,
        seed=123,
        standardize_obs=True,
        steps_per_obs_stats_increment=5,
    ):
        self.policy = policy
        self.seed = seed
        self.processes = []
        self.selector = selectors.DefaultSelector()

        self.next_obs = []
        self.current_obs = []

        self.current_pids = []
        self.average_reward = None
        self.cumulative_timesteps = 0
        self.min_inference_size = min_inference_size

        self.standardize_obs = standardize_obs
        self.steps_per_obs_stats_increment = steps_per_obs_stats_increment
        self.steps_since_obs_stats_update = 0
        self.obs_stats = None

        self.ep_rews = []
        self.trajectory_map = []
        self.prev_time = 0
        self.completed_trajectories = []

        self.n_procs = 0
        import struct

        self.packed_header = comm_consts.pack_message(comm_consts.POLICY_ACTIONS_HEADER)

    def collect_timesteps(self, n):
        """
        Collect a specified number of timesteps from the environment.

        :param n: Number of timesteps to collect.
        :return: A tuple containing the collected data arrays and additional information.
                - states (np.ndarray): Array of states.
                - actions (np.ndarray): Array of actions.
                - log_probs (np.ndarray): Array of log probabilities of actions.
                - rewards (np.ndarray): Array of rewards.
                - next_states (np.ndarray): Array of next states.
                - dones (np.ndarray): Array of done flags.
                - truncated (np.ndarray): Array of truncated flags.
                - n_collected (int): Number of timesteps collected.
                - elapsed_time (float): Time taken to collect the timesteps.
        """

        t1 = time.perf_counter()
        states = []
        actions = []
        log_probs = []
        rewards = []
        next_states = []
        dones = []
        truncated = []

        n_collected = 0
        n_procs = max(1, len(self.processes))
        n_obs_per_inference = min(self.min_inference_size, n_procs)
        collected_metrics = []
        # Collect n timesteps.
        while n_collected < n:
            # Send actions for the current observations and collect new states to act on. Note that the next states
            # will not necessarily be from the environments that we just sent actions to. Whatever timestep data happens
            # to be lying around in the buffer will be collected and used in the next inference step.

            self._send_actions()
            (
                collected_metrics_this_pass,
                n_collected_this_pass,
            ) = self._collect_responses(n_obs_per_inference)
            n_collected += n_collected_this_pass
            collected_metrics += collected_metrics_this_pass

            for proc_id in self.current_pids:
                if self.next_obs[proc_id] is not None:
                    self.current_obs[proc_id] = self.next_obs[proc_id]
                    self.next_obs[proc_id] = None

            self._sync_trajectories()

        # Organize our new timesteps into the appropriate lists.
        for proc_id, trajectory in enumerate(self.trajectory_map):
            self.completed_trajectories.append(trajectory)
            self.trajectory_map[proc_id] = BatchedTrajectory()

        for trajectory in self.completed_trajectories:
            trajectories = trajectory.get_all()
            if len(trajectories) == 0:
                continue

            for traj in trajectories:
                (
                    trajectory_states,
                    trajectory_actions,
                    trajectory_log_probs,
                    trajectory_rewards,
                    trajectory_next_states,
                    trajectory_dones,
                ) = traj
                trajectory_truncated = [0 for _ in range(len(trajectory_dones))]
                trajectory_truncated[-1] = 1 if trajectory_dones[-1] == 0 else 0
                states += trajectory_states
                actions += trajectory_actions
                log_probs += trajectory_log_probs
                rewards += trajectory_rewards
                next_states += trajectory_next_states
                dones += trajectory_dones
                truncated += trajectory_truncated

        self.cumulative_timesteps += n_collected
        t2 = time.perf_counter()
        self.completed_trajectories = []

        return (
            (
                np.asarray(states),
                np.asarray(actions),
                np.asarray(log_probs),
                np.asarray(rewards),
                np.asarray(next_states),
                np.asarray(dones),
                np.asarray(truncated),
            ),
            collected_metrics,
            n_collected,
            t2 - t1,
        )

    def _sync_trajectories(self):
        for proc_id, trajectory in enumerate(self.trajectory_map):
            if trajectory.update():
                self.completed_trajectories.append(trajectory)
                self.trajectory_map[proc_id] = BatchedTrajectory()

    @torch.no_grad()
    def _send_actions(self):
        """
        Send actions to environment processes based on current observations.
        """
        if len(self.current_pids) == 0:
            return

        dimensions = []
        obs, pids = [], []
        for proc_id in self.current_pids:
            o = self.current_obs[proc_id]
            if o is None:
                continue

            obs.append(o)
            pids.append(proc_id)
            dimensions.append(o.shape[0])

        if not dimensions:
            return

        inference_batch = np.concatenate(obs, axis=0)
        actions, log_probs = self.policy.get_action(inference_batch)
        actions = actions.numpy().astype(np.float32)

        step = 0
        for proc_id, dim_0 in zip(pids, dimensions):
            process, parent_end, child_endpoint, shm_view = self.processes[proc_id]
            stop = step + dim_0

            state = inference_batch[step:stop]
            action = actions[step:stop]
            logp = log_probs[step:stop]
            parent_end.sendto(self.packed_header + action.tobytes(), child_endpoint)
            self.trajectory_map[proc_id].action = action
            self.trajectory_map[proc_id].log_prob = logp
            self.trajectory_map[proc_id].state = state

            step += dim_0

        self.current_pids = []

    def _collect_responses(self, n_obs_per_inference):
        """
        Collect responses from environment processes and update trajectory data.
        :return: Number of responses collected.
        """

        n_collected = 0
        self.current_pids = []
        collected_metrics = []

        if self.standardize_obs:
            obs_mean = self.obs_stats.mean[0]
            obs_std = self.obs_stats.std[0]
        else:
            obs_mean = None
            obs_std = None

        while n_collected < n_obs_per_inference:
            for key, event in self.selector.select():
                if not (event & selectors.EVENT_READ):
                    continue

                parent_end, fd, events, proc_id = key
                process, parent_end, child_endpoint, shm_view = self.processes[proc_id]
                n_collected += self._collect_response(
                    proc_id, parent_end, shm_view, collected_metrics, obs_mean, obs_std
                )

        return collected_metrics, n_collected

    def _collect_response(
        self, proc_id, parent_end, shm_view, collected_metrics, obs_mean, obs_std
    ):
        available_data = parent_end.recv(PACKET_MAX_SIZE)
        header = frombuffer(available_data, dtype=np.float32, count=comm_consts.HEADER_LEN)

        if header[0] != comm_consts.ENV_STEP_DATA_HEADER[0]:
            return 0

        prev_n_agents = int(shm_view[0])
        done = shm_view[1]
        n_elements_in_state_shape = int(shm_view[2])

        metrics_len = int(shm_view[3])
        if metrics_len != 0:
            metrics_shape = [int(d) for d in shm_view[4 : 4 + metrics_len]]
            n_metrics = prod(metrics_shape)
        else:
            metrics_shape = (0,)
            n_metrics = 0

        state_shape_start = 4 + metrics_len
        if n_elements_in_state_shape == 1:
            state_shape = [1, int(shm_view[state_shape_start])]
        else:
            state_shape = [
                int(arg)
                for arg in shm_view[
                    state_shape_start : state_shape_start
                    + n_elements_in_state_shape
                ]
            ]

        rew_start = state_shape_start + n_elements_in_state_shape
        rew_end = rew_start + prev_n_agents

        shm_shapes = self.shm_shapes[proc_id]
        if shm_shapes is None or shm_shapes != (metrics_shape, state_shape):
            self.shm_shapes[proc_id] = (metrics_shape, state_shape)
            rews = np.reshape(shm_view[rew_start : rew_end], (rew_end - rew_start,))
            metrics = np.reshape(shm_view[rew_end : rew_end + n_metrics], metrics_shape)
            obs = np.reshape(shm_view[rew_end + n_metrics : rew_end + n_metrics + prod(state_shape)], state_shape)
            self.shm_cache[proc_id] = (rews, metrics, obs)

        rews, metrics, next_observation = self.shm_cache[proc_id]

        collected_metrics.append(metrics)

        if self.standardize_obs:
            if (
                self.steps_since_obs_stats_update
                > self.steps_per_obs_stats_increment
            ):
                self.obs_stats.increment(next_observation, state_shape[0])
                self.steps_since_obs_stats_update = 0
            else:
                self.steps_since_obs_stats_update += 1

            next_observation = np.clip(
                (next_observation - obs_mean) / obs_std, a_min=-5, a_max=5
            )

        if prev_n_agents > 1:
            n_collected = prev_n_agents
            for i in range(prev_n_agents):
                if i >= len(self.ep_rews[proc_id]):
                    self.ep_rews[proc_id].append(rews[i])
                else:
                    self.ep_rews[proc_id][i] += rews[i]
        else:
            n_collected = 1
            rews = rews[0]
            self.ep_rews[proc_id][0] += rews

        if done:
            if self.average_reward is None:
                self.average_reward = self.ep_rews[proc_id][0]
            else:
                for ep_rew in self.ep_rews[proc_id]:
                    self.average_reward = self.average_reward * 0.9 + ep_rew * 0.1

            self.ep_rews[proc_id] = [0]

        if proc_id not in self.current_pids:
            self.current_pids.append(proc_id)

        self.next_obs[proc_id] = next_observation
        self.trajectory_map[proc_id].reward = rews
        self.trajectory_map[proc_id].next_state = next_observation
        self.trajectory_map[proc_id].done = done

        if state_shape[0] != prev_n_agents:
            self.completed_trajectories.append(self.trajectory_map[proc_id])
            self.trajectory_map[proc_id] = BatchedTrajectory()

        return n_collected

    def _get_initial_states(self):
        """
        Retrieve initial states from environment processes.
        :return: None.
        """

        self.current_pids = []
        for proc_id, proc_package in enumerate(self.processes):
            process, parent_end, child_endpoint, shm_view = proc_package

            available_data = parent_end.recv(PACKET_MAX_SIZE)
            message = comm_consts.unpack_message(available_data)

            header = message[: comm_consts.HEADER_LEN]
            if header == comm_consts.ENV_RESET_STATE_HEADER:
                self.current_pids.append(proc_id)

                data = message[comm_consts.HEADER_LEN :]

                n_elements_in_state_shape = int(data[0])
                shape = [int(arg) for arg in data[1 : 1 + n_elements_in_state_shape]]
                if n_elements_in_state_shape == 1:
                    shape = [1, shape[0]]

                obs = np.reshape(data[1 + n_elements_in_state_shape :], shape)
                if self.standardize_obs:
                    if self.obs_stats is None:
                        self.obs_stats = WelfordRunningStat(shape=shape[-1])
                    self.obs_stats.increment(obs, shape[0])

                # self.current_obs.append(obs)
                self.current_obs[proc_id] = obs

    def _get_env_shapes(self):
        """
        Retrieve environment observation and action space shapes from one of the connected environment processes.
        :return: A tuple containing observation shape, action shape, and action space type.
        """
        process, parent_end, child_endpoint, shm_view = self.processes[0]
        request_msg = comm_consts.pack_message(comm_consts.ENV_SHAPES_HEADER)
        parent_end.sendto(request_msg, child_endpoint)

        obs_shape, action_shape, action_space_type = -1, -1, -1
        done = False

        while not done:
            available_data = parent_end.recv(PACKET_MAX_SIZE)
            message = comm_consts.unpack_message(available_data)
            header = message[: comm_consts.HEADER_LEN]
            if header == comm_consts.ENV_SHAPES_HEADER:
                data = message[comm_consts.HEADER_LEN :]

                obs_shape, action_shape, action_space_type = data
                done = True

        return int(obs_shape), int(action_shape), int(action_space_type)

    def init_processes(
        self,
        n_processes,
        build_env_fn,
        collect_metrics_fn=None,
        spawn_delay=None,
        render=False,
        render_delay: Union[float, None] = None,
        shm_buffer_size = 8192
    ):
        """
        Initialize and spawn environment processes.
        :param n_processes: Number of processes to spawn.
        :param build_env_fn: A function to build the environment for each process.
        :param collect_metrics_fn: A user-defined function that the environment processes will use to collect metrics
               about the environment at each timestep.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing observation shape, action shape, and action space type.
        """

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)
        self.n_procs = n_processes

        import multiprocessing.sharedctypes
        self.shm_size = shm_buffer_size // 4
        self.shm_buffer = multiprocessing.sharedctypes.RawArray('f', n_processes * self.shm_size)
        self.shm_shapes = [None for i in range(n_processes)]
        self.shm_cache = [None for i in range(n_processes)]
        self.processes = [None for i in range(n_processes)]
        self.ep_rews = [[0] for i in range(n_processes)]
        self.trajectory_map = [BatchedTrajectory() for i in range(n_processes)]
        self.current_obs = [None for i in range(n_processes)]
        self.next_obs = [None for i in range(n_processes)]

        # Spawn child processes
        for proc_id in tqdm(range(n_processes)):
            render_this_proc = proc_id == 0 and render

            # Create socket to communicate with child
            parent_end = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            parent_end.bind(("127.0.0.1", 0))

            shm_offset = proc_id * self.shm_size * 4

            process = context.Process(
                target=batched_agent_process,
                args=(
                    proc_id,
                    parent_end.getsockname(),
                    self.shm_buffer,
                    shm_offset,
                    self.shm_size,
                    self.seed + proc_id,
                    render_this_proc,
                    render_delay,
                ),
            )
            process.start()

            shm_view = frombuffer(buffer=self.shm_buffer, dtype=np.float32, offset=shm_offset, count=self.shm_size)

            self.processes[proc_id] = (process, parent_end, shm_view)

            self.selector.register(parent_end, selectors.EVENT_READ, proc_id)

        # Initialize child processes
        for proc_id in range(n_processes):
            process, parent_end, shm_view = self.processes[proc_id]

            # Get child endpoint
            _, child_endpoint = parent_end.recvfrom(1)

            if spawn_delay is not None:
                time.sleep(spawn_delay)

            p = pickle.dumps(("initialization_data", build_env_fn, collect_metrics_fn))
            parent_end.sendto(p, child_endpoint)

            self.processes[proc_id] = (process, parent_end, child_endpoint, shm_view)

        self._get_initial_states()
        return self._get_env_shapes()

    def cleanup(self):
        """
        Clean up resources and terminate processes.
        """
        import traceback

        for proc_id, proc_package in enumerate(self.processes):
            process, parent_end, child_endpoint, shm_view = proc_package

            try:
                parent_end.sendto(
                    comm_consts.pack_message(comm_consts.STOP_MESSAGE_HEADER),
                    child_endpoint,
                )
            except Exception:
                print("Unable to join process")
                traceback.print_exc()
                print("Failed to send stop signal to child process!")
                traceback.print_exc()

            try:
                process.join()
            except Exception:
                print("Unable to join process")
                traceback.print_exc()

            try:
                parent_end.close()
            except Exception:
                print("Unable to close parent connection")
                traceback.print_exc()
