from itertools import product, combinations
from utils.scenario_generator import generate_scenario
import numpy as np
import os

def enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes=True):
    """
    Enumerates all possible split configurations.
    
    Args:
        allowed_splits (list): Safe layer boundaries (sorted list).
        num_nodes (int): Number of computation nodes.
        allow_empty_nodes (bool): If True, nodes may have zero layers.

    Returns:
        list of tuples: Each element is a split configuration
                                [(node_id, start, end), ...]
    """
    action_indices = {}
    allowed_splits = sorted(allowed_splits)
    K = len(allowed_splits)
    assert K >= 2, "Need at least start/end boundaries"

    # Internal boundary index space
    if allow_empty_nodes:
        cut_idx_space = range(1, K)       # allow final boundary index for empty nodes
    else:
        cut_idx_space = range(1, K - 1)   # exclude final boundary from cuts

    # Generate all cut index combinations
    if allow_empty_nodes:
        # combinations with replacement
        all_cuts = product(cut_idx_space, repeat=num_nodes - 1)
    else:
        # unique combinations only
        all_cuts = combinations(cut_idx_space, num_nodes - 1)

    actions = []
    for cuts in all_cuts:
        cuts = sorted(cuts)
        points_idx = [0] + list(cuts) + [K - 1]

        # Ensure UE gets at least 1 layer
        if points_idx[1] == points_idx[0]:
            points_idx[1] = min(points_idx[0] + 1, K - 1)

        # Build split config
        splits = []
        for node_id in range(num_nodes):
            s_idx = points_idx[node_id]
            e_idx = points_idx[node_id + 1]
            start = int(allowed_splits[s_idx])
            end = int(allowed_splits[e_idx])
            splits.append((node_id, start, end))

        # ensure no duplicates
        if splits not in actions:
            actions.append(splits)

    for i, a in enumerate(actions):
        #print(a)
        action_indices[i] = a

    return actions, action_indices

def extended_action_space(splits, compression_rates):
    """
    Enumerates all combinations of splits and compression rates.
    Args:
        splits (list): List containing all (unique) possible split options.
        compression_rates (list): List containing the configured compression rates.

    Returns:
        Dict of the format {'split': selected_split_config, 'compression': compression_rate} of the full action space
        and corresponding indices
    """
    full_action_space = []
    action_indices_full = {}
    ue_only_split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
    # extended action space, where each action is in the format:
    # action = {'split': selected_split_config, 'compression': compression_rate}
    for selected_split_config in splits:
        for compression in compression_rates:
            # in case of ue only computation, select only 1.0 for the value of rho
            if selected_split_config == ue_only_split and compression < 1.0:
                continue
            else:
                full_action_space.append({'split': selected_split_config, 'compression': compression})

    for idx, action in enumerate(full_action_space):
        #print(a)
        action_indices_full[idx] = action
    return full_action_space, action_indices_full


#Sample usage:
if __name__ == "__main__":
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    params = generate_scenario()
    compression_rate_list = params['compression_rates']
    allowed_splits = [0, 3, 6, 10, 14, 18]
    num_nodes = 4
    all_actions, action_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes=True)
    print(f"Total actions: {len(all_actions)}")
    for i, a in enumerate(all_actions):
        print(a)
        action_indices[i] = a
    #print(all_actions[26])
    #print(action_indices)
    act = [(0, 0, 3), (1, 3, 3), (2, 3, 6), (3, 6, 18)]
    for k, v in action_indices.items():
        if v == act:
            print(k)
    action_space, _ = extended_action_space(all_actions, compression_rate_list)
    print(action_space)
    print(len(action_space))

