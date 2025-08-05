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

def generate_random_split(allowed_splits, num_nodes, allow_empty_nodes=True):
    """
    Args:
        allowed_splits (list): Layer indices where splitting is safe without model refactoring
                               (e.g., [0, 3, 6, 10, 14, 18]).
        num_nodes (int): Number of computation nodes to split the model across.
    
    Returns:
        list: A list of tuples (node_id, start_layer, end_layer) for each node.
    """
    # Randomly pick internal split points
    split_points = sorted(
        np.random.choice(allowed_splits[1:-1], num_nodes - 1, replace=False)
    )
    split_points = [allowed_splits[0]] + split_points + [allowed_splits[-1]]

    # If empty nodes allowed, duplicate some split points
    if allow_empty_nodes:
        raw_points = np.random.choice(split_points, num_nodes + 1, replace=True)
        split_points = np.sort(raw_points)

    splits = []
    last_end = allowed_splits[0]

    for i in range(num_nodes):
        start = int(last_end)
        end = int(split_points[i + 1])

        # Node 0 must have at least one layer
        if i == 0 and start == end and end < allowed_splits[-1]:
            end += 1

        splits.append((i, start, end))
        last_end = end  # Move forward (no backtracking)

    # Force last active node to cover remaining layers
    for idx in reversed(range(num_nodes)):
        if splits[idx][1] != splits[idx][2]:
            splits[idx] = (splits[idx][0], splits[idx][1], allowed_splits[-1])
            break

    return splits