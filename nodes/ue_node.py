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

    def compute(self, model, x, start_layer, end_layer, flops):
        """
        Execute the assigned subset of layers for inference on the UE.

        Args:
            model (nn.Module): The full model.
            x (Tensor): Input tensor or intermediate activation.
            start_layer (int): Start index of assigned layers.
            end_layer (int): End index of assigned layers.
            flops (float): FLOPs for the assigned segment.

        Returns:
            Tuple: (output tensor, computation time in seconds, energy in joules)
        """
        if start_layer == end_layer:
            # No layers assigned to this node
            return x, 0.0, 0.0

        start = time.time()
        output = model.forward(x, start_layer, end_layer)
        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        energy = calculate_energy(comp_time, self.power)

        return output, comp_time, energy
