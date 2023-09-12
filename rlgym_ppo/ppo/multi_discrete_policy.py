"""
File: multi_discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes 8 discrete distributions over the actions
    available in Rocket League.
"""

import torch.nn as nn
import torch
import numpy as np
from rlgym_ppo.util import torch_functions


class MultiDiscreteFF(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        self.layer_sizes = layer_sizes
        bins = [3,3,3,3,3,2,2,2]
        n_output_nodes = sum(bins)
        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_output_nodes))
        self.model = nn.Sequential(*layers).to(self.device)
        self.splits = bins
        self.multi_discrete = torch_functions.MultiDiscreteRolv(bins)

    def clone(self, to: str = None):
        device = self.device if to is None else to
        cloned_policy = MultiDiscreteFF(self.input_shape, self.layer_sizes, device)
        cloned_policy.load_state_dict(self.state_dict())
        if to is not None:
            cloned_policy.to(to)

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

        logits = self.get_output(obs)

        #TODO not sure how to do this better - very slow atm
        if deterministic:
            start = 0
            action = []
            for split in self.splits:
                action.append(logits[..., start:start+split].argmax(dim=-1))
                start += split
            action = torch.stack(action).cpu().numpy()
            return action, 1

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

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
        logits = self.get_output(obs)

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        entropy = distribution.entropy().to(self.device)
        log_probs = distribution.log_prob(acts).to(self.device)

        return log_probs, entropy.mean()
