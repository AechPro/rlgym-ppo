def batched_agent_process(proc_id, endpoint, shm_buffer, shm_offset, shm_size, seed, render, render_delay):
    """
    Function to interact with an environment and communicate with the learner through a pipe.

    :param proc_id: Process id
    :param endpoint: Parent endpoint for communication
    :param shm_buffer: Shared memory buffer
    :param shm_offset: Shared memory offset
    :param shm_size: Shared memory size
    :param seed: Seed for environment and action space randomization.
    :param render: Whether the environment will be rendered every timestep.
    :param render_delay: Amount of time in seconds to delay between steps while rendering.
    :return: None
    """

    import pickle
    import socket
    import time

    import gym
    import numpy as np

    from rlgym_ppo.batched_agents import comm_consts

    if render:
        try:
            from rlviser_py import get_game_paused, get_game_speed
        except ImportError:
            def get_game_speed() -> float:
                return 1.0

            def get_game_paused() -> bool:
                return False

    def _append_array(array, offset, data):
        size = data.size if isinstance(data, np.ndarray) else len(data)
        end = offset + size
        array[offset:end] = data[:]
        return end

    env = None
    metrics_encoding_function = None
    shm_view = None
    shm_shapes = None

    POLICY_ACTIONS_HEADER = comm_consts.POLICY_ACTIONS_HEADER
    ENV_SHAPES_HEADER = comm_consts.ENV_SHAPES_HEADER
    STOP_MESSAGE_HEADER = comm_consts.STOP_MESSAGE_HEADER

    PACKED_ENV_STEP_DATA_HEADER = comm_consts.pack_message(
        comm_consts.ENV_STEP_DATA_HEADER
    )
    header_len = comm_consts.HEADER_LEN

    # Create a socket and send dummy data to tell parent our endpoint
    pipe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pipe.bind(("127.0.0.1", 0))
    pipe.sendto(b"0", endpoint)

    # Wait for initialization data from the learner.
    while env is None:
        data = pickle.loads(pipe.recv(4096))
        if data[0] == "initialization_data":
            build_env_fn = data[1]
            metrics_encoding_function = data[2]

            env = build_env_fn()

    # Seed everything.
    env.action_space.seed(seed)
    reset_state = env.reset()

    if type(reset_state) != np.ndarray:
        reset_state = np.asarray(reset_state, dtype=np.float32)
    elif reset_state.dtype != np.float32:
        reset_state = reset_state.astype(np.float32)

    state_shape = [float(arg) for arg in np.shape(reset_state)]
    n_elements_in_state_shape = float(len(state_shape))
    n_agents = state_shape[0] if n_elements_in_state_shape > 1 else 1
    obs_buffer = reset_state.tobytes()

    message_floats = (
        comm_consts.ENV_RESET_STATE_HEADER + [n_elements_in_state_shape] + state_shape
    )
    packed_message_floats = comm_consts.pack_message(message_floats) + obs_buffer
    pipe.sendto(packed_message_floats, endpoint)

    action_buffer = None
    action_slice_size = 0

    frombuffer = np.frombuffer

    prev_n_agents = n_agents

    # Primary interaction loop.
    try:
        last_render_time = time.time()
        render_time_compensation = 0
        while True:
            message_bytes = pipe.recv(4096)
            message = frombuffer(message_bytes, dtype=np.float32)
            # message = byte_headers.unpack_message(message_bytes)
            header = message[:header_len]

            if header[0] == POLICY_ACTIONS_HEADER[0]:
                prev_n_agents = n_agents
                data = message[header_len:]

                if action_buffer is None:
                    action_buffer = np.reshape(data, (int(n_agents), -1)).copy()
                    action_slice_size = action_buffer.shape[1]
                else:
                    for i in range(int(n_agents)):
                        action_buffer[i] = data[
                            i * action_slice_size : (i + 1) * action_slice_size
                        ]

                # print("got actions", action_buffer.shape,"|",n_agents,"|",prev_n_agents)
                step_data = env.step(action_buffer)
                if len(step_data) == 4:
                    truncated = False
                    obs, rew, done, info = step_data
                else:
                    obs, rew, done, truncated, info = step_data

                if n_agents == 1:
                    rew = [float(rew)]

                if done or truncated:
                    obs = np.asarray(env.reset(), dtype=np.float32)
                    n_agents = float(obs.shape[0]) if len(obs.shape) > 1 else 1

                    state_shape = [float(arg) for arg in obs.shape]
                    n_elements_in_state_shape = len(state_shape)

                    action_buffer = np.zeros((int(n_agents), action_buffer.shape[-1]))

                if type(obs) != np.ndarray:
                    obs = np.asarray(obs, dtype=np.float32)
                elif obs.dtype != np.float32:
                    obs = obs.astype(np.float32)

                done = 1.0 if done else 0.0
                truncated = 1.0 if truncated else 0.0

                if metrics_encoding_function is not None:
                    metrics = metrics_encoding_function(info["state"])
                    metrics_shape = [float(arg) for arg in metrics.shape]
                else:
                    metrics = np.empty(shape=(0,))
                    metrics_shape = []

                if shm_view is None or shm_shapes != (prev_n_agents, n_agents):
                    shm_shapes = (prev_n_agents, n_agents)
                    count = 5 + len(metrics_shape) + len(state_shape) + len(rew) + metrics.size + obs.size
                    assert(count <= shm_size), "ATTEMPTED TO CREATE AGENT MESSAGE BUFFER LARGER THAN MAXIMUM ALLOWED SIZE"
                    shm_view = np.frombuffer(buffer=shm_buffer, dtype=np.float32, offset=shm_offset, count=count)

                offset = _append_array(shm_view, 0, [prev_n_agents, done, truncated, n_elements_in_state_shape, len(metrics_shape)])
                offset = _append_array(shm_view, offset, metrics_shape)
                offset = _append_array(shm_view, offset, state_shape)
                offset = _append_array(shm_view, offset, rew)
                offset = _append_array(shm_view, offset, metrics)
                offset = _append_array(shm_view, offset, obs.flatten())

                pipe.sendto(PACKED_ENV_STEP_DATA_HEADER, endpoint)

                if render:
                    env.render()
                    if render_delay is not None:
                        render_delta_time = time.time() - last_render_time
                        last_render_time = time.time()

                        target_delay = render_delay / get_game_speed()
                        render_time_compensation = np.clip(
                            render_time_compensation + (target_delay - render_delta_time),
                            -target_delay, 0)
                        sleep_delay = max(0, target_delay + render_time_compensation)
                        time.sleep(sleep_delay)

                    while get_game_paused():
                        time.sleep(0.1)

            elif header[0] == ENV_SHAPES_HEADER[0]:
                t = type(env.action_space)
                action_space_type = 0.0  # "discrete"
                action_type = "Discrete"
                if t == gym.spaces.multi_discrete.MultiDiscrete:
                    action_space_type = 1.0  # "multi-discrete"
                    action_type = "Multi-discrete"
                elif t == gym.spaces.box.Box:
                    action_space_type = 2.0  # "continuous"
                    action_type = "Continuous"

                if hasattr(env.action_space, "n"):
                    n_acts = float(env.action_space.n)
                else:
                    n_acts = float(np.prod(env.action_space.shape))

                # Print out the environment shapes and action space type.
                print("Received request for env shapes, returning:")
                print(F"- Observations shape: {env.observation_space.shape}")
                print(F"- Number of actions: {n_acts}")
                print(F"- Action space type: {action_space_type} ({action_type})")
                print("--------------------")

                env_shape = float(np.prod(env.observation_space.shape))
                message_floats = ENV_SHAPES_HEADER + [
                    env_shape,
                    n_acts,
                    action_space_type,
                ]
                pipe.sendto(comm_consts.pack_message(message_floats), endpoint)

            elif header[0] == STOP_MESSAGE_HEADER[0]:
                break

    except Exception:
        import traceback

        print("ERROR IN BATCHED AGENT LOOP")
        traceback.print_exc()

    finally:
        pipe.close()
        env.close()
