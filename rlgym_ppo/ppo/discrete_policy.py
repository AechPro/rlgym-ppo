"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""


from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        for size in layer_sizes[1:]:
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        policy_output = self.model(obs)
        return policy_output

    def get_action(self, obs, deterministic=False):
        """
        Function to the an action and the log of its probability for an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """

        probs = self.get_output(obs)
        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        distribution = Categorical(probs=probs)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts):
        """
        Function to compute the data necessary for backpropagation.
        :param obs: Observations to pass through the policy.
        :param acts: Actions taken by the policy.
        :return: Action log probs & entropy.
        """
        acts = acts.flatten()
        probs = self.get_output(obs)
        distribution = Categorical(probs=probs)
        entropy = distribution.entropy()
        log_probs = distribution.log_prob(acts)

        return log_probs.to(self.device), entropy.to(self.device).mean()