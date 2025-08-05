"""
comm_utils.py

Utility functions for modeling communication latency and energy consumption 
"""

def calculate_comm_time(data_size_bytes, bandwidth_MBps):

    """
    Args:
        data_size_bytes (float): Size of the data to be transmitted (in bytes).
        bandwidth_MBps (float): Available bandwidth (in megabytes per second).

    Returns:
        float: Communication time in seconds.
    """
    return data_size_bytes / (bandwidth_MBps * 1e6)

def calculate_comm_energy(data_size_bytes, energy_cost_per_byte):
    return data_size_bytes * energy_cost_per_byte
