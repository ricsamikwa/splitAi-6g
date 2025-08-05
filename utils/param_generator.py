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
    freqs = np.random.uniform(2.5, 3.5, num_nodes)  # GHz
    flops_per_cycle = np.random.uniform(4, 8, num_nodes)
    bandwidth = np.random.uniform(100, 500, num_nodes)  # MB/s
    return freqs, flops_per_cycle, bandwidth
