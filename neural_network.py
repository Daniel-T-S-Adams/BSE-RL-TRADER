import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from typing import Iterable


class Network(nn.Module):
    """
    A fully connected neural network class with 
    specified dimensions and output activation.

    Attributes:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        network (nn.Module): The sequential neural network created from the specified dimensions.
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None):
        """
        Initializes the Network with the given dimensions and output activation.

        Args:
            dims (Iterable[int]): A list or tuple of integers specifying the sizes of each layer in the network.
                                  The first element is the input size, and the last element is the output size.
            output_activation (nn.Module): The activation function to apply to the output layer. Default is None.
        """
        super().__init__()
        self.input_size = dims[0]
        self.output_size = dims[-1]
        self.network = self.make_seq(dims, output_activation)

    
    def make_seq(self, dims: Iterable[int], output_activation: nn.Module) -> nn.Module:
        """
        Creates a sequential neural network from the specified dimensions and output activation.

        Args:
            dims (Iterable[int]): A list or tuple of integers specifying the sizes of each layer in the network.
            output_activation (nn.Module): The activation function to apply to the output layer.

        Returns:
            nn.Module: A sequential neural network constructed from the specified layers and activation functions.
        """
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if output_activation:
            layers.append(output_activation)
        return nn.Sequential(*layers)
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes a forward pass through the network.

        Args:
            x: The input tensor to the network.

        Returns:
            The output of the network after passing through the sequential layers.
        """
        return self.network(x)
    

# # Example usage
# state_dim = 100  # Example state space size
# action_dim = 20  # Example action space size (number of possible actions)
# hidden_layers = [128, 64]  # Example hidden layers

# # Create an instance of the Network
# policy_net = Network([state_dim] + [action_dim], nn.Softmax(dim=-1))

# # Generate a random state (for example purposes)
# state = torch.FloatTensor([1.0, 0.5, -0.1, 0.0, 1.2, -0.7, 0.3, 0.8, -0.5, 0.6])

# # Get the action probabilities from the policy network
# action_probs = policy_net(state)

# # Sample an action based on the probabilities
# action = torch.multinomial(action_probs, 1).item()

# print("Action probabilities:", action_probs)
# print("Selected action:", action)

