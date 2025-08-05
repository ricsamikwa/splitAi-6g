import torch
import time
from utils.flop_utils import calculate_inference_time
from utils.energy_utils import calculate_energy

class UENode:
    def __init__(self, cpu_freq, flops_per_cycle, power):
        self.cpu_freq = cpu_freq
        self.flops_per_cycle = flops_per_cycle
        self.power = power
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, model, x, start_layer, end_layer, flops):
        start = time.time()
        output = model.forward(x, start_layer, end_layer)
        comp_time = calculate_inference_time(flops, self.cpu_freq, self.flops_per_cycle)
        energy = calculate_energy(comp_time, self.power)
        return output, comp_time, energy
