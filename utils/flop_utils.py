"""
flop_utils.py

Utility functions for estimating inference time based on the number of 
floating-point operations (FLOPs) and CPU hardware characteristics.
"""

def calculate_inference_time(flops, cpu_freq, flops_per_cycle):

    """
    Estimate the computation time required for inference on a given device.

    Args:
        flops (float): Total number of floating-point operations required by the submodel.
        cpu_freq (float): CPU frequency of the device (in GHz).
        flops_per_cycle (float): Number of FLOPs that can be executed per CPU cycle.

    Returns:
        float: Estimated inference time (in seconds).
    """
    return flops / (cpu_freq * 1e9 * flops_per_cycle)
