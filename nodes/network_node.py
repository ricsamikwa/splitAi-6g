import torch
from utils.flop_utils import calculate_inference_time

class NetworkNode:
    def __init__(self, node_id, cpu_freq, flops_per_cycle):
        """
        Represents a network compute node
        """
        self.node_id = node_id
        self.cpu_freq = cpu_freq
        self.flops_per_cycle = flops_per_cycle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, model, x, start_layer, end_layer, flops):
        """
        Executes inference for layers assigned to this node.
        """
        output = model.forward(x, start_layer, end_layer)
        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        return output, comp_time
