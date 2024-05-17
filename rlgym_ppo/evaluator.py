import json
import os
import random
from typing import Callable, Tuple, Union

import gym
import numpy as np
import torch

from rlgym_ppo.batched_agents import BatchedAgentManager
from rlgym_ppo.ppo import PPOLearner
from rlgym_ppo.util import KBHit, WelfordRunningStat


class Evaluator(object):
    def __init__(
        # fmt: off
        self,
        env_create_function: Callable[..., gym.Env],
        render: bool = True,
        render_delay: float = 0,

        timestep_limit: int = 5_000_000_000,
        standardize_obs: bool = True,
        max_returns_per_stats_increment: int = 150,
        steps_per_obs_stats_increment: int = 5,

        policy_layer_sizes: Tuple[int, ...] = (256, 256, 256),
        critic_layer_sizes: Tuple[int, ...] = (256, 256, 256),
        continuous_var_range: Tuple[float, ...] = (0.1, 1.0),

        checkpoint_load_folder: Union[str, None] = None,
        instance_launch_delay: Union[float, None] = None,

        random_seed: int = 123,
        shm_buffer_size: int = 8192,
        device: str = "cpu",
    ):

        assert (
            env_create_function is not None
        ), "MUST PROVIDE A FUNCTION TO CREATE RLGYM FUNCTIONS TO INITIALIZE RLGYM-PPO"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.max_returns_per_stats_increment = max_returns_per_stats_increment

        if device in {"auto", "gpu"} and torch.cuda.is_available():
            self.device = "cuda:0"
            torch.backends.cudnn.benchmark = True
        elif device == "auto" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        print(f"Using device {self.device}")
        self.timestep_limit = timestep_limit

        print("Initializing processes...")
        self.agent = BatchedAgentManager(
            None,
            min_inference_size=1,
            seed=random_seed,
            standardize_obs=standardize_obs,
            steps_per_obs_stats_increment=steps_per_obs_stats_increment,
        )
        obs_space_size, act_space_size, action_space_type = self.agent.init_processes(
            n_processes=1,
            build_env_fn=env_create_function,
            collect_metrics_fn=None,
            spawn_delay=instance_launch_delay,
            render=render,
            render_delay=render_delay,
            shm_buffer_size=shm_buffer_size,
        )
        obs_space_size = np.prod(obs_space_size)

        print("Initializing PPO...")
        self.ppo_learner = PPOLearner(
            obs_space_size,
            act_space_size,
            device=self.device,
            batch_size=1,
            mini_batch_size=1,
            n_epochs=1,
            continuous_var_range=continuous_var_range,
            policy_type=action_space_type,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_lr=1,
            critic_lr=1,
            clip_range=1,
            ent_coef=1,
        )

        self.agent.policy = self.ppo_learner.policy

        assert checkpoint_load_folder is not None, "MUST PROVIDE A CHECKPOINT FOLDER"
        self.load(checkpoint_load_folder)

        print("Learner successfully initialized!")

    def eval(self):
        try:
            self._eval()
        except Exception:
            import traceback

            print("\n\nLEARNING LOOP ENCOUNTERED AN ERROR\n")
            traceback.print_exc()
        finally:
            self.cleanup()

    def _eval(self):
        # Class to watch for keyboard hits
        kb = KBHit()
        print("Press (p) to pause, (q) quit\n")

        # While the number of timesteps we have collected so far is less than the
        # amount we are allowed to collect.
        while self.agent.cumulative_timesteps < self.timestep_limit:
            # Collect the desired number of timesteps from our agent.
            self.agent.collect_timesteps(120)

            if "cuda" in self.device:
                torch.cuda.empty_cache()

            # Check if keyboard press
            # p: pause, any key to resume
            # q: quit

            if kb.kbhit():
                c = kb.getch()
                if c == "p":  # pause
                    print("Paused, press any key to resume")
                    while True:
                        if kb.kbhit():
                            break
                if c == "q":
                    return
                if c == "p":
                    print("Resuming...\n")

    def load(self, folder_path):
        """
        Function to load the learning algorithm from a checkpoint.

        :param folder_path: Path to the checkpoint folder that will be loaded.
        :return: None
        """

        # Make sure the folder exists.
        assert os.path.exists(folder_path), f"UNABLE TO LOCATE FOLDER {folder_path}"
        print(f"Loading from checkpoint at {folder_path}")

        # Load stuff.
        self.ppo_learner.load_from(folder_path)

        with open(os.path.join(folder_path, "BOOK_KEEPING_VARS.json"), "r") as f:
            book_keeping_vars = dict(json.load(f))
            self.agent.cumulative_timesteps = book_keeping_vars["cumulative_timesteps"]
            self.agent.average_reward = book_keeping_vars["policy_average_reward"]
            self.ppo_learner.cumulative_model_updates = book_keeping_vars[
                "cumulative_model_updates"
            ]

            if (
                self.agent.standardize_obs
                and "obs_running_stats" in book_keeping_vars.keys()
            ):
                self.agent.obs_stats = WelfordRunningStat(1)
                self.agent.obs_stats.from_json(book_keeping_vars["obs_running_stats"])

        print("Checkpoint loaded!")
        return False

    def cleanup(self):
        """
        Function to clean everything up before shutting down.
        :return: None.
        """

        if type(self.agent) == BatchedAgentManager:
            self.agent.cleanup()
