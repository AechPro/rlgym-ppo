"""
File: value_estimator.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which models the value function of a policy.
"""
import torch.nn as nn
import torch


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        for size in layer_sizes[1:]:
            layers.append(nn.Linear(size, size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return self.model(x)