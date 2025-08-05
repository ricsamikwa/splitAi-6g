"""
ue_node.py

Defines the UE node responsible for computing a subset of layers in a split DNN inference setting.
"""

import torch
import time
from utils.flop_utils import calculate_inference_time
from utils.energy_utils import calculate_energy

class UENode:
    def __init__(self, cpu_freq, flops_per_cycle, power):
        """
        Args:
            cpu_freq (float): CPU frequency in GHz.
            flops_per_cycle (float): Number of FLOPs executed per CPU cycle.
            power (float): Power consumption in watts.
        """
        self.cpu_freq = cpu_freq
        self.flops_per_cycle = flops_per_cycle
        self.power = power
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, model, x, start_layer, end_layer, flops, include_fc=False):
        """
        Execute the assigned subset of layers for inference on the UE.

        Args:
            model (nn.Module): The model.
            x (Tensor): Input tensor or intermediate activation.
            start_layer (int): Start index of assigned layers.
            end_layer (int): End index of assigned layers.
            flops (float): FLOPs for the assigned segment.
            include_fc (bool): If True, also execute fully connected layers.

        Returns:
            Tuple: (output tensor, computation time in seconds, energy in joules)
        """
        if start_layer == end_layer and not include_fc:
            # No layers assigned to this node
            return x, 0.0, 0.0

        start = time.time()
        output = x

        # Run convolutional layers if assigned
        if start_layer < end_layer:
            output = model.forward(output, start_layer, end_layer)

        # If all conv layers are on UE, run FC layers as well
        if include_fc:
            output = torch.flatten(output, 1)
            output = model.fc1(output)
            output = model.fc2(output)
            output = model.fc3(output)

        elapsed_time = time.time() - start
        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        energy = calculate_energy(comp_time, self.power)

        return output, comp_time, energy
