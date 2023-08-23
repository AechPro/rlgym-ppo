def batched_agent_process(pipe, seed, render, render_delay: float):
    """
    Function to interact with an environment and communicate with the learner through a pipe.

    :param pipe: A bidirectional communication pipe.
    :param seed: Seed for environment and action space randomization.
    :param render: Whether the environment will be rendered every timestep.
    :param render_delay: Amount of time in seconds to delay between steps while rendering.
    :return: None
    """

    import time
    import gym
    env = None

    # Wait for initialization data from the learner.
    while env is None:
        data = pipe.recv()
        if data[0] == "initialization_data":
            build_env_fn = data[1]

            env = build_env_fn()

    # Seed everything.
    env.action_space.seed(seed)
    reset_state = env.reset()
    pipe.send(("reset_state", reset_state))

    # Primary interaction loop.
    try:
        while True:
            data = pipe.recv()

            # If we have an action, take a step in the environment and transmit the timestep data back to the learner.
            if data[0] == "action":
                action = data[1]
                obs, rew, done, _ = env.step(action)
                if done:
                    obs = env.reset()

                done = 1 if done else 0

                pipe.send(("env_step_data", (obs, rew, done)))
                if render:
                    env.render()
                    if render_delay != 0:
                        time.sleep(render_delay)

            # If the learner is requesting obs and action space shapes, provide them.
            elif data[0] == "get_env_shapes":
                t = type(env.action_space)
                action_space_type = "discrete"
                if t == gym.spaces.multi_discrete.MultiDiscrete:
                    action_space_type = "multi-discrete"
                elif t == gym.spaces.box.Box:
                    action_space_type = "continuous"

                if hasattr(env.action_space, "n"):
                    n_acts = env.action_space.n
                else:
                    n_acts = env.action_space.shape

                print("Received request for env shapes, returning",env.observation_space.shape, n_acts, action_space_type)
                pipe.send(("env_shapes", (env.observation_space.shape, n_acts, action_space_type)))

            elif data[0] == "stop":
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