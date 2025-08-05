import numpy as np

def generate_random_split(total_layers, num_nodes):
    """
    Generate a random contiguous split of model layers across nodes.
    Ensures each node gets at least 1 layer.
    """
    split_points = sorted(np.random.choice(range(1, total_layers), num_nodes - 1, replace=False))
    split_points = [0] + split_points + [total_layers]

    splits = []
    for i in range(num_nodes):
        start = split_points[i]
        end = split_points[i+1] - 1
        splits.append((i, start, end))
    return splits
