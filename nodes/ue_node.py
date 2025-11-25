"""
ue_node.py

Defines the UE node responsible for computing a subset of layers in a split DNN inference setting.
Optionally applies channel compression (ρ) on the activation that is offloaded to the network.
"""

import torch
import time
from utils.flop_utils import calculate_inference_time
from utils.energy_utils import calculate_energy

class UENode:
    def __init__(self, cpu_freq, flops_per_cycle, power, rho: float = 1.0):
        """
        Args:
            cpu_freq (float): CPU frequency in GHz.
            flops_per_cycle (float): Number of FLOPs executed per CPU cycle.
            power (float): Power consumption in watts.
            rho (float): Channel compression ratio in (0, 1]. Default 1.0
                         means no compression.
        """
        self.cpu_freq = cpu_freq
        self.flops_per_cycle = flops_per_cycle
        self.power = power
        self.rho = rho
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _compress_feature(feat: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Channel-reduction compression:
        keep the first ⌊ρC⌋ channels of the feature map.

        Args:
            feat (Tensor): Feature map of shape (B, C, H, W).
            rho (float): Compression ratio in (0, 1].

        Returns:
            Compressed feature map of shape (B, C_red, H, W), where
            C_red = max(1, int(rho * C)).
        """
        if rho >= 1.0:
            return feat

        B, C, H, W = feat.shape
        C_red = max(1, int(rho * C))

        feat_red = feat[:, :C_red, :, :]
        return feat_red
    
    def compute(self, model, x, start_layer, end_layer, flops, include_fc=False, rho: float = None):
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

        if rho is None:
            rho = self.rho

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

        else:
            # UE offloads → apply compression
            output = self._compress_feature(output, rho)

        elapsed_time = time.time() - start
        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        energy = calculate_energy(comp_time, self.power)

        return output, comp_time, energy
