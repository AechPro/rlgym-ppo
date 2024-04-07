"""
File: torch_functions.py
Author: Matthew Allen

Description:
    A helper file for misc. PyTorch functions.

"""


import torch.nn as nn
import torch


class MapContinuousToAction(nn.Module):
    """
    A class for policies using the continuous action space. Continuous policies output N*2 values for N actions where
    each value is in the range [-1, 1]. Half of these values will be used as the mean of a multi-variate normal distribution
    and the other half will be used as the diagonal of the covariance matrix for that distribution. Since variance must
    be positive, this class will map the range [-1, 1] for those values to the desired range (defaults to [0.1, 1]) using
    a simple linear transform.
    """
    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()

        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        # map the right half of x from [-1, 1] to [range_min, range_max].
        return x[..., :n], x[..., n:] * self.m + self.b


def compute_gae(rews, dones, truncated, values, gamma=0.99, lmbda=0.95, return_std=1):
    """
    Function to estimate the advantage function for a series of states and actions using the
    general advantage estimator (GAE).

    :param rews: List of rewards.
    :param dones: List of done signals.
    :param truncated: List of truncated signals.
    :param values: List of value function estimates.
    :param gamma: Gamma hyper-parameter.
    :param lmbda: Lambda hyper-parameter.
    :param return_std: Standard deviation of the returns (used for reward normalization).
    :return: Bootstrapped value function estimates, GAE results, returns.
    """
    next_values = values[1:]

    last_gae_lam = 0
    n_returns = len(rews)
    adv = [0 for _ in range(n_returns)]
    returns = [0 for _ in range(n_returns)]
    last_return = 0

    for step in reversed(range(n_returns)):
        not_done = 1 - dones[step]
        not_trunc = 1 - truncated[step]

        if return_std is not None:
            norm_rew = min(max(rews[step] / return_std, -10), 10)
        else:
            norm_rew = rews[step]

        pred_ret = norm_rew + gamma * next_values[step] * not_done
        delta = pred_ret - values[step]
        ret = rews[step] + last_return * gamma * not_done * not_trunc
        returns[step] = ret
        last_return = ret
        last_gae_lam = delta + gamma * lmbda * not_done * not_trunc * last_gae_lam
        adv[step] = last_gae_lam


    advantages = torch.as_tensor(adv, dtype=torch.float32)
    values = torch.as_tensor([v + a for v, a in zip(values[:-1], adv)], dtype=torch.float32)
    return values, advantages, returns


class MultiDiscreteRolv(nn.Module):
    """
    A class to handle the multi-discrete action space in Rocket League. There are 8 potential actions, 5 of which can be
    any of {-1, 0, 1} and 3 of which can be either of {0, 1}. This class takes 21 logits, appends -inf to the final 3
    such that each of the 8 actions has 3 options (to avoid a ragged list), then builds a categorical distribution over
    each class for each action. Credit to Rolv Arild for coming up with this method.
    """
    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins

    def make_distribution(self, logits):
        """
        Function to make the multi-discrete categorical distribution for a group of logits.
        :param logits: Logits which parameterize the distribution.
        :return: None.
        """

        # Split the 21 logits into the expected bins.
        logits = torch.split(logits, self.bins, dim=-1)

        # Separate triplets from the split logits.
        triplets = torch.stack(logits[:5], dim=-1)

        # Separate duets and pad the final dimension with -inf to create triplets.
        duets = torch.nn.functional.pad(torch.stack(logits[5:], dim=-1), pad=(0,0,0,1), value=float("-inf"))

        # Un-split the logits now that the duets have been converted into triplets and reshape them into the correct shape.
        logits = torch.cat((triplets, duets), dim=-1).swapdims(-1, -2)

        # Construct a distribution with our fixed logits.
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) # Unsure about this sum operation.
