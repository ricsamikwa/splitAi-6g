"""
network_node.py

Defines a network node responsible for executing a portion of the model
during split inference across multiple computation nodes. Optionally
decompresses the activation coming from the UE if channel compression
(ρ < 1) was applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.flop_utils import calculate_inference_time

class NetworkNode:
    def __init__(self, node_id, cpu_freq, flops_per_cycle, rho: float = 1.0):
        """
        Args:
            node_id (int): Unique identifier for the node.
            cpu_freq (float): CPU frequency in GHz.
            flops_per_cycle (float): Number of FLOPs per CPU cycle.
        """
        self.node_id = node_id
        self.cpu_freq = cpu_freq
        self.rho = rho
        self.flops_per_cycle = flops_per_cycle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @staticmethod
    def _find_in_channels(block: nn.Module):
        """Return in_channels of the first Conv2d inside a block (Sequential or Conv2d)."""
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                return m.in_channels
        return None

    @staticmethod
    def _find_last_conv_out_channels(model: nn.Module):
        """Return out_channels of the last Conv2d in the conv stack."""
        for m in reversed(list(model.conv_layers.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        return None

    @staticmethod
    def _decompress_for_conv(feat: torch.Tensor, rho: float, model, start_layer: int):
        """
        Decompress feature map before running conv layers.

        We restore the original channel dimension expected by the first conv
        layer executed on this node (in_channels of that conv).
        """
        if rho is None or rho >= 1.0:
            return feat  # no compression → nothing to do

        if feat.dim() != 4:
            return feat  # only compress/decompress 4D feature maps

        B, C_red, H, W = feat.shape

        # Get the conv block at this start_layer
        block = list(model.conv_layers.children())[start_layer]
        C_target = NetworkNode._find_in_channels(block)
        if C_target is None:
            # Should not happen with your VGG, but safe fallback
            return feat

        if C_red == C_target:
            return feat
        elif C_red > C_target:
            return feat[:, :C_target, :, :]
        else:
            pad_ch = C_target - C_red
            pad = (0, 0, 0, 0, 0, pad_ch)  # (W_left, W_right, H_left, H_right, C_left, C_right)
            return F.pad(feat, pad)

    @staticmethod
    def _decompress_for_fc(feat: torch.Tensor, rho: float, model):
        """
        Decompress feature map when this node runs ONLY the FC layers (no conv).
        We restore to the out_channels of the LAST Conv2d (e.g., 512 for VGG16).
        """
        if rho is None or rho >= 1.0:
            return feat

        if feat.dim() != 4:
            return feat

        B, C_red, H, W = feat.shape
        C_target = NetworkNode._find_last_conv_out_channels(model)
        if C_target is None:
            return feat

        if C_red == C_target:
            return feat
        elif C_red > C_target:
            return feat[:, :C_target, :, :]
        else:
            pad_ch = C_target - C_red
            pad = (0, 0, 0, 0, 0, pad_ch)
            return F.pad(feat, pad)

        
    def compute(self, model, x, start_layer, end_layer, flops, include_fc=False, rho: float = None):
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
        # Backwards compatibility: old calls remain valid.
        if rho is None:
            rho = self.rho

        # Case 1: no conv layers assigned to this node
        if start_layer == end_layer:
            if not include_fc:
                # Purely idle node
                return x, 0.0

            # FC-only node: may receive compressed feature from UE
            output = self._decompress_for_fc(x, rho, model)

            output = torch.flatten(output, 1)
            output = model.fc1(output)
            output = model.fc2(output)
            output = model.fc3(output)

            comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
            return output, comp_time

        # Case 2: node has conv layers (and optionally FC)
        output = x

        # Decompress before running convs, if needed
        output = self._decompress_for_conv(output, rho, model, start_layer)

        # Run conv layers
        output = model.forward(output, start_layer, end_layer)

        # Optionally run FC layers if this is the last node
        if include_fc:
            output = torch.flatten(output, 1)
            output = model.fc1(output)
            output = model.fc2(output)
            output = model.fc3(output)

        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        return output, comp_time