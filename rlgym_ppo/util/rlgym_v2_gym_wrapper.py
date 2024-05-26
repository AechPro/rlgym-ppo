import gym
import numpy as np


class RLGymV2GymWrapper(object):
    def __init__(self, rlgym_env):
        self.rlgym_env = rlgym_env
        self.agent_map = {}
        self.obs_buffer = np.zeros(1)
        print('WARNING: CALLING ENV.RESET() ONE EXTRA TIME TO DETERMINE STATE AND ACTION SPACES')
        obs_dict = rlgym_env.reset()
        obs_list = list(obs_dict.values())
        act_space = list(rlgym_env.action_spaces.values())[0][1]
        obs_space = list(rlgym_env.observation_spaces.values())[0][1]

        self.is_discrete = False
        if type(act_space) == int:
            self.action_space = gym.spaces.Discrete(n=act_space)
            self.is_discrete = True
        else:
            self.action_space = None

        if type(obs_space) == int and obs_space > 0:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space, ))
        else:
            if obs_list:
                self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=np.shape(obs_list[0]))
            else:
                self.observation_space = None

    def reset(self):
        self.agent_map.clear()
        obs_dict = self.rlgym_env.reset()
        idx = 0
        obs_vec = []

        for agent_id, agent_obs in obs_dict.items():
            self.agent_map[idx] = agent_id
            obs_vec.append(agent_obs)
            idx += 1

        self.obs_buffer = np.asarray(obs_vec)
        return self.obs_buffer

    def step(self, actions):
        if self.is_discrete:
            actions = actions.astype(np.int32)

        action_dict = {}
        for i in range(len(actions)):
            agent_id = self.agent_map[i]
            action = actions[i]
            action_dict[agent_id] = action

        obs_dict, reward_dict, terminated_dict, truncated_dict = self.rlgym_env.step(action_dict)

        rews = []
        done = False
        truncated = False
        idx = 0
        for agent_id, agent_obs in obs_dict.items():
            self.obs_buffer[idx] = agent_obs
            rews.append(reward_dict[agent_id])
            done = done or terminated_dict[agent_id]
            truncated = truncated or truncated_dict[agent_id]
            idx += 1

        info = {"state": self.rlgym_env.state}
        return self.obs_buffer, rews, done, truncated, info

    def render(self):
        self.rlgym_env.render()

    def seed(self, seed):
        pass

    def close(self):
        self.rlgym_env.close()