"""
energy_utils.py

Utility functions for modeling energy consumption of computation nodes 
(UE or network nodes) during model inference
"""

def calculate_energy(time, power):
    return time * power
