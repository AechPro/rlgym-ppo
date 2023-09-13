"""
File: value_estimator.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which models the value function of a policy.
"""
import torch.nn as nn
import torch
import numpy as np


class ValueEstimator(nn.Module):
    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers).to(self.device)
    
    
    def clone(self, to: str = None):
        device = self.device if to is None else to
        cloned_value_estimator = ValueEstimator(self.input_shape, self.layer_sizes, device)
        cloned_value_estimator.load_state_dict(self.state_dict())
        if to is not None:
            cloned_value_estimator.to(to)

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return self.model(x)
