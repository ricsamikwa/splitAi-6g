"""
split_generator.py

Utility function for generating random split points for model partitioning.
Ensures that each node gets a contiguous set of layers without splitting inside
VGG blocks (avoiding channel mismatches).

Note:
    Some nodes may not be allocated any layers (start_layer == end_layer),
    allowing flexible configurations where fewer than num_nodes actively 
    compute layers.
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
    # Generate standard split points
    split_points = sorted(
        np.random.choice(allowed_splits[1:-1], num_nodes - 1, replace=False)
    )
    split_points = [allowed_splits[0]] + split_points + [allowed_splits[-1]]

    # If empty nodes are allowed, duplicate some split points randomly
    if allow_empty_nodes:
        split_points = np.sort(
            np.random.choice(split_points, num_nodes + 1, replace=True)
        )

    splits = []
    for i in range(num_nodes):
        start = int(split_points[i])
        end = int(split_points[i + 1])
        splits.append((i, start, end))
    return splits
