"""
network_node.py

Defines a network node responsible for executing a portion of the model
during split inference across multiple computation nodes.
"""

import torch
from utils.flop_utils import calculate_inference_time

class NetworkNode:
    def __init__(self, node_id, cpu_freq, flops_per_cycle):
        """
        Args:
            node_id (int): Unique identifier for the node.
            cpu_freq (float): CPU frequency in GHz.
            flops_per_cycle (float): Number of FLOPs per CPU cycle.
        """
        self.node_id = node_id
        self.cpu_freq = cpu_freq
        self.flops_per_cycle = flops_per_cycle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, model, x, start_layer, end_layer, flops, include_fc=False):
        """
        Execute assigned model layers for inference.

        Args:
            model (nn.Module): The model.
            x (Tensor): Input tensor or intermediate activation.
            start_layer (int): Start index of assigned layers.
            end_layer (int): End index of assigned layers.
            flops (float): FLOPs for the assigned segment.
            include_fc (bool): If True, also execute fully connected layers
                               (typically for the last network node).

        Returns:
            Tuple: (output tensor, computation time in seconds)
        """
        if start_layer == end_layer and not include_fc:
            return x, 0.0

        output = x

        # Run convolutional layers
        if start_layer < end_layer:
            output = model.forward(output, start_layer, end_layer)

        # Optionally run FC layers if this is the last node
        if include_fc:
            output = torch.flatten(output, 1)
            output = model.fc1(output)
            output = model.fc2(output)
            output = model.fc3(output)

        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        return output, comp_time