from itertools import product, combinations
import numpy as np

def enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes=True):
    """
    Enumerates all possible split configurations.
    
    Args:
        allowed_splits (list): Safe layer boundaries (sorted list).
        num_nodes (int): Number of computation nodes.
        allow_empty_nodes (bool): If True, nodes may have zero layers.

    Returns:
        list of list of tuples: Each element is a split configuration 
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


#Sample usage:
if __name__ == "__main__":
    allowed_splits = [0, 3, 6, 10, 14, 18]
    num_nodes = 4
    action_indices = {}
    all_actions = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes=True)
    print(f"Total actions: {len(all_actions)}")
    for i, a in enumerate(all_actions):
        #print(a)
        action_indices[i] = a
    print(all_actions)
    #print(action_indices)
    act = [(0, 0, 3), (1, 3, 3), (2, 3, 6), (3, 6, 18)]
    for k, v in action_indices.items():
        if v == act:
            print(k)

