import numpy as np

allowed_splits = [0, 5, 10, 17, 24, 31]  
# Explanation:
# 0 → start of model
# 5 → after block1
# 10 → after block2
# 17 → after block3
# 24 → after block4
# 31 → after block5 (end of conv_layers)

import numpy as np

def generate_safe_random_split(allowed_splits, num_nodes):
    """
    Generate safe random split points from predefined allowed indices.
    Ensures each node gets at least one full block of layers.
    """
    # choose (num_nodes - 1) internal split points (can't pick first or last)
    split_points = sorted(np.random.choice(allowed_splits[1:-1], num_nodes - 1, replace=False))
    split_points = [allowed_splits[0]] + split_points + [allowed_splits[-1]]

    splits = []
    for i in range(num_nodes):
        start = split_points[i]
        end = split_points[i+1]
        splits.append((i, start, end))
    return splits
