"""
param_generator.py

Utility functions for generating random hardware parameters for 
computation nodes 
"""

import numpy as np
import pandas as pd
import os

from utils.scenario_generator import generate_scenario
from utils.logging_utils import parse_episode_number, return_order

order_to_convert = 1000
order = return_order(order_to_convert)

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
    ue_freq = np.random.uniform(1.8, 2.8)        # GHz
    #ue_freq = 2.0
    ue_flops_per_cycle = np.random.uniform(1.5, 3.0)
    #ue_flops_per_cycle = 2.0
    ue_bandwidth = np.random.uniform(100, 400)    # MB/s (UE link speed)
    #ue_bandwidth = 300

    # Network nodes specs
    freqs = np.random.uniform(3.0, 4.5, num_nodes - 1)      # GHz
    #freqs = [5.0 for _ in range(num_nodes - 1)]
    flops_per_cycle = np.random.uniform(5.0, 9.0, num_nodes - 1)
    #flops_per_cycle = [7.0 for _ in range(num_nodes - 1)]
    bandwidth = np.random.uniform(500, 2000, num_nodes - 1) # MB/s (net-to-net)
    #bandwidth = [500 for _ in range(num_nodes - 1)]

    return ue_freq, ue_flops_per_cycle, ue_bandwidth, freqs, flops_per_cycle, bandwidth

def write_params_to_file():
    params = generate_scenario()

    num_nodes = 4

    n_episodes = params['n_episodes']
    # change this parameter manually
    episode_duration = params['episode_duration']
    start_episode = params['start_episode']
    time_interval = params['time_interval']

    # save headers in a list
    headers = ['ue_freq', 'ue_flops_per_cycle', 'ue_bandwidth']
    for k in range(1, num_nodes):
        headers.append('freqs{}'.format(k))
        headers.append('flops_per_cycle{}'.format(k))
        headers.append('bandwidth{}'.format(k))
    #print(headers)
    headers_and_params_per_episode = []
    for ep in range(start_episode, n_episodes + 1):
        params_per_episode = []
        ep_str = parse_episode_number(order, ep)
        for k in range(0, episode_duration, time_interval):
            params_per_timestep = []
            ue_freq, ue_flops_per_cycle, ue_bandwidth, freqs, flops_per_cycle, bandwidth = generate_params(num_nodes)
            # append each param to list
            params_per_timestep.append(ue_freq)
            params_per_timestep.append(ue_flops_per_cycle)
            params_per_timestep.append(ue_bandwidth)
            for i in range(num_nodes - 1):
                params_per_timestep.append(freqs[i])
                params_per_timestep.append(flops_per_cycle[i])
                params_per_timestep.append(bandwidth[i])
            params_per_episode.append(params_per_timestep)
        #print(params_per_episode)
        # convert to dataframe
        df = pd.DataFrame(params_per_episode, columns=headers)
        # save to file
        df.to_csv('input/episode_parameters/ep_0001.csv')


def read_params_from_file(episode, num_nodes):
    # save headers in a list
    headers = ['ue_freq', 'ue_flops_per_cycle', 'ue_bandwidth']
    for k in range(1, num_nodes):
        headers.append('freqs{}'.format(k))
        headers.append('flops_per_cycle{}'.format(k))
        headers.append('bandwidth{}'.format(k))
    ep_str = parse_episode_number(order, episode)
    path = 'input/episode_parameters/ep_{}.csv'.format(ep_str)
    df = pd.read_csv(path)
    #print(df['ue_freq'][0])
    return df



if __name__ == '__main__':
    #read_params_from_file(episode=1, num_nodes=4)
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    write_params_to_file()