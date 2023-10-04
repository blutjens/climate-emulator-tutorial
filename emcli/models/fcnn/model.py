"""
Fully Connected Neural networks (with residual layers
"""

import numpy as np

import torch
import torch.nn as nn

class ResNet(nn.Module):
    """
    ResNet block
    Source: https://stackoverflow.com/questions/57229054/how-to-implement-my-own-resnet-with-torch-nn-sequential-in-pytorch
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class FCNN(nn.Module):
    # Residual FCNN
    # https://stackoverflow.com/questions/46141690/how-to-write-a-pytorch-sequential-model
    def __init__(self, dim_in, dim_out, n_layers=2, n_units=128, 
        resNet=True, n_res_blocks=1):
        """
        Args:
            dim_in int: Number of input channels
            dim_out int: Number of outputs
            n_layers int: Number of hidden layers
            n_units int: Number of hidden units per hidden layers
            dtype
        """
        super(FCNN, self).__init__()
        
        self.act = nn.ReLU#ReLU

        layers = []
        # Input layer
        layers.append(nn.Linear(dim_in, n_units))
        layers.append(self.act())
        if resNet:
            # Hidden Residual layers
            for r in range(n_res_blocks):
                for l in range(n_layers):
                    res_layers = []
                    res_layers.append(nn.Linear(n_units, n_units))
                    res_layers.append(self.act())
                layers.append(ResNet(nn.Sequential(*res_layers)))            
        else:
            # Hidden layers
            for l in range(n_layers):
                layers.append(nn.Linear(n_units, n_units))
                layers.append(self.act())
        
        # Output layer
        layers.append(nn.Linear(n_units, dim_out))
        self.net = nn.Sequential(*layers)

    def predict(self, test_inputs):
        x = self.net(test_inputs)
        return (x)

    def forward(self, inputs):
        x = self.net(inputs)
        return (x)

