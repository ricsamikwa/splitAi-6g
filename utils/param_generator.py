"""
param_generator.py

Utility functions for generating random hardware parameters for 
computation nodes 
"""

import numpy as np

def generate_params(num_nodes):

    """
    Args:
        num_nodes (int): Number of computation nodes (e.g., UE + network nodes).

    Returns:
        tuple:
            freqs (ndarray): CPU frequencies for each node (in GHz)
            flops_per_cycle (ndarray): FLOPs that can be executed per CPU cycle for each node
            bandwidth (ndarray): Available bandwidth between consecutive nodes (in MB/s)
    """
    # UE-specific specs
    #ue_freq = np.random.uniform(1.8, 2.8)        # GHz
    ue_freq = 2.0
    #ue_flops_per_cycle = np.random.uniform(1.5, 3.0)
    ue_flops_per_cycle = 2.0
    #ue_bandwidth = np.random.uniform(100, 400)    # MB/s (UE link speed)
    ue_bandwidth = 300

    # Network nodes specs
    #freqs = np.random.uniform(3.0, 4.5, num_nodes - 1)      # GHz
    freqs = [3.0 for _ in range(num_nodes - 1)]
    #flops_per_cycle = np.random.uniform(5.0, 9.0, num_nodes - 1)
    flops_per_cycle = [7.0 for _ in range(num_nodes - 1)]
    #bandwidth = np.random.uniform(500, 2000, num_nodes - 1) # MB/s (net-to-net)
    bandwidth = [500 for _ in range(num_nodes - 1)]

    return ue_freq, ue_flops_per_cycle, ue_bandwidth, freqs, flops_per_cycle, bandwidth