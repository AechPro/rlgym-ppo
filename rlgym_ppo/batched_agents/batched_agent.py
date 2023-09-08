

def batched_agent_process(proc_id, endpoint, seed, render, render_delay):
    """
    Function to interact with an environment and communicate with the learner through a pipe.

    :param endpoint: Parent endpoint for communication
    :param seed: Seed for environment and action space randomization.
    :param render: Whether the environment will be rendered every timestep.
    :param render_delay: Amount of time in seconds to delay between steps while rendering.
    :return: None
    """

    import time
    import gym
    import numpy as np
    import pickle
    import socket
    import struct
    from rlgym_ppo.batched_agents import comm_consts
    env = None
    metrics_encoding_function = None

    POLICY_ACTIONS_HEADER = comm_consts.POLICY_ACTIONS_HEADER
    ENV_SHAPES_HEADER = comm_consts.ENV_SHAPES_HEADER
    STOP_MESSAGE_HEADER = comm_consts.STOP_MESSAGE_HEADER

    PACKED_ENV_STEP_DATA_HEADER = comm_consts.pack_message(comm_consts.ENV_STEP_DATA_HEADER)
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

    message_floats = comm_consts.ENV_RESET_STATE_HEADER + [n_elements_in_state_shape] + state_shape
    packed_message_floats = comm_consts.pack_message(message_floats) + obs_buffer
    pipe.sendto(packed_message_floats, endpoint)

    action_buffer = None
    action_slice_size = 0

    pack = struct.pack
    frombuffer = np.frombuffer

    prev_n_agents = n_agents

    # Primary interaction loop.
    try:
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
                        action_buffer[i] = data[i*action_slice_size:(i+1)*action_slice_size]

                # print("got actions", action_buffer.shape,"|",n_agents,"|",prev_n_agents)
                obs, rew, done, info = env.step(action_buffer)

                if n_agents == 1:
                    rew = [float(rew)]

                if done:
                    obs = np.asarray(env.reset(), dtype=np.float32)
                    n_agents = float(obs.shape[0]) if len(obs.shape) > 1 else 1

                    state_shape = [float(arg) for arg in obs.shape]
                    n_elements_in_state_shape = len(state_shape)

                    action_buffer = np.zeros((int(n_agents), action_buffer.shape[-1]))

                if type(obs) != np.ndarray:
                    obs = np.asarray(obs, dtype=np.float32)
                elif obs.dtype != np.float32:
                    obs = obs.astype(np.float32)

                done = 1. if done else 0.

                obs_buffer = obs.tobytes()

                if metrics_encoding_function is not None:
                    metrics = metrics_encoding_function(info["state"])
                    metrics_shape = [float(arg) for arg in metrics.shape]
                    metrics_bytes = metrics.tobytes()

                    message_floats = [prev_n_agents, done, n_elements_in_state_shape, len(metrics_shape)] + metrics_shape + state_shape + rew
                    packed = pack("%sf" % len(message_floats), *message_floats)
                    message_bytes = PACKED_ENV_STEP_DATA_HEADER + packed + metrics_bytes + obs_buffer

                else:
                    message_floats = [prev_n_agents, done, n_elements_in_state_shape, 0] + state_shape + rew
                    packed = pack("%sf" % len(message_floats), *message_floats)
                    message_bytes = PACKED_ENV_STEP_DATA_HEADER + packed + obs_buffer

                pipe.sendto(message_bytes, endpoint)
                if render:
                    env.render()
                    if render_delay is not None:
                        time.sleep(render_delay)

            elif header[0] == ENV_SHAPES_HEADER[0]:
                t = type(env.action_space)
                action_space_type = 0.  # "discrete"
                if t == gym.spaces.multi_discrete.MultiDiscrete:
                    action_space_type = 1.  # "multi-discrete"
                elif t == gym.spaces.box.Box:
                    action_space_type = 2.  # "continuous"

                if hasattr(env.action_space, "n"):
                    n_acts = float(env.action_space.n)
                else:
                    n_acts = float(np.prod(env.action_space.shape))

                print("Received request for env shapes, returning", env.observation_space.shape, n_acts,
                      action_space_type)

                env_shape = float(np.prod(env.observation_space.shape))
                message_floats = ENV_SHAPES_HEADER + [env_shape, n_acts, action_space_type]
                pipe.sendto(pack("%sf" % len(message_floats), *message_floats), endpoint)

            elif header[0] == STOP_MESSAGE_HEADER[0]:
                break

    # Catch everything and print it.
    except:
        import traceback
        print("ERROR IN BATCHED AGENT LOOP")
        traceback.print_exc()

    # Close the pipe and local environment instance.
    finally:
        pipe.close()
        env.close()
