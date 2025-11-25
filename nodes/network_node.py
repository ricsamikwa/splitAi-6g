"""
network_node.py

Defines a network node responsible for executing a portion of the model
during split inference across multiple computation nodes. Optionally
decompresses the activation coming from the UE if channel compression
(ρ < 1) was applied.
"""

import torch
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
    def _decompress_feature(
        feat: torch.Tensor,
        rho: float,
        model,
        start_layer: int,
    ) -> torch.Tensor:
        """
        Decompress feature map that was compressed by channel reduction.

        Restore the original channel dimension expected by the first
        conv layer executed on this node. 
        Args:
            feat (Tensor): Compressed feature of shape (B, C_red, H, W).
            rho (float): Compression ratio used at UE.
            model (nn.Module): Model containing conv_layers.
            start_layer (int): Index of the first conv layer this node runs.

        Returns:
            Tensor: Decompressed feature of shape (B, C_target, H, W).
        """
        if rho >= 1.0:
            return feat

        B, C_red, H, W = feat.shape

        conv_layer = list(model.conv_layers.children())[start_layer]
        C_target = conv_layer.in_channels

        if C_red == C_target:
            return feat
        elif C_red > C_target:
            # Too many channels: just keep the first C_target.
            return feat[:, :C_target, :, :]
        else:
            # Too few channels: zero-pad the missing ones.
            pad_channels = C_target - C_red
            # F.pad pads in the order (W_left, W_right, H_left, H_right, C_left, C_right)
            pad = (0, 0, 0, 0, 0, pad_channels)
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

        if start_layer == end_layer and not include_fc:
            return x, 0.0

        output = x

        # Run convolutional layers
        if start_layer < end_layer:
            output = self._decompress_feature(output, rho, model, start_layer)
            output = model.forward(output, start_layer, end_layer)

        # Optionally run FC layers if this is the last node
        if include_fc:
            output = torch.flatten(output, 1)
            output = model.fc1(output)
            output = model.fc2(output)
            output = model.fc3(output)

        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        return output, comp_time