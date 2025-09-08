"""
split_generator.py

Utility function for generating random split points for model partitioning.
Ensures that each node gets a contiguous set of layers without splitting inside
VGG blocks (avoiding channel mismatches).

Note:
    Some nodes may not be allocated any layers (start_layer == end_layer),
    allowing flexible configurations where fewer than num_nodes actively 
    compute layers.

    Node 0 (UE) is always assigned at least one layer to ensure proper
    handling of the raw input image.
"""

import numpy as np
from action_space import enumerate_action_space

def generate_random_split(allowed_splits, num_nodes, allow_empty_nodes=True):
    """
    Args:
        allowed_splits (list): Layer indices where splitting is safe without model refactoring
                               (e.g., [0, 3, 6, 10, 14, 18]).
        num_nodes (int): Number of computation nodes to split the model across.
    
    Returns:
        list: A list of tuples (node_id, start_layer, end_layer) for each node.
    """
    # Build full action space once
    actions, _ = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
    
    # Sample one action uniformly
    idx = np.random.randint(len(actions))

    return actions[idx]

    # # Randomly pick internal split points
    # allowed_splits = sorted(allowed_splits)
    # K = len(allowed_splits)
    # assert K >= 2, "Need at least start/end boundaries"

    # # >>> FIX: allow choosing K-1 (final boundary) when empty nodes are allowed
    # if allow_empty_nodes:
    #     cut_idx_space = np.arange(1, K)      # 1..K-1  (enables 'all on UE')
    # else:
    #     cut_idx_space = np.arange(1, K - 1)  # 1..K-2  (classic internal cuts only)

    # # Choose cut indices
    # if cut_idx_space.size == 0:
    #     points_idx = np.array([0, K - 1])
    # else:
    #     if allow_empty_nodes:
    #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=True)
    #     else:
    #         if num_nodes - 1 > cut_idx_space.size:
    #             raise ValueError("Not enough unique split points for all nodes.")
    #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=False)
    #     cuts_idx = np.sort(cuts_idx)
    #     points_idx = np.concatenate(([0], cuts_idx, [K - 1]))

    # # Ensure UE (first segment) has at least one layer
    # if points_idx[1] == points_idx[0]:
    #     points_idx[1] = min(points_idx[0] + 1, K - 1)

    # # Ensure non-decreasing sequence
    # for i in range(2, len(points_idx)):
    #     if points_idx[i] < points_idx[i - 1]:
    #         points_idx[i] = points_idx[i - 1]

    # # Build splits
    # splits = []
    # for node_id in range(num_nodes):
    #     s_idx = points_idx[node_id]
    #     e_idx = points_idx[node_id + 1]
    #     start = int(allowed_splits[s_idx])
    #     end = int(allowed_splits[e_idx])
    #     splits.append((node_id, start, end))

    # return splits
    

if __name__ == "__main__":
    allowed_splits = [0, 3, 6, 10, 14, 18]
    num_nodes = 4

    print("Testing random split sampling...")
    for _ in range(5):
        split = generate_random_split(allowed_splits, num_nodes, allow_empty_nodes=True)
        print(split)
    