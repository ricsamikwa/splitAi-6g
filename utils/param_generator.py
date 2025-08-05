import numpy as np

def generate_params(num_nodes):
    freqs = np.random.uniform(1.0, 3.0, num_nodes)  # GHz
    flops_per_cycle = np.random.uniform(2, 5, num_nodes)
    bandwidth = np.random.uniform(50, 200, (num_nodes, num_nodes))  # MB/s
    return freqs, flops_per_cycle, bandwidth
