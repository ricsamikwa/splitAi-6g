import torch
import torch.nn as nn

def compute_flops_per_layer(model, input_size=(3, 224, 224)):
    """
    Compute FLOPs for each convolutional and fully connected layer in VGG16.

    Args:
        model (nn.Module): VGG16 model instance.
        input_size (tuple): Input image size (C, H, W).

    Returns:
        dict: Mapping of layer indices and FC layer names to FLOPs.
    """
    flops_per_layer = {}
    #x = torch.randn(1, *input_size)
    x = torch.randn(1, *input_size).to(next(model.parameters()).device)  # <-- minimal line fix to GPU

    # Fully flatten the conv layers
    def flatten_layers(module):
        flat = []
        for layer in module.children():
            if isinstance(layer, nn.Sequential):
                flat.extend(flatten_layers(layer))
            else:
                flat.append(layer)
        return flat

    flat_layers = flatten_layers(model.conv_layers)
    conv_idx = 0

    for i, layer in enumerate(flat_layers):
        if isinstance(layer, nn.Conv2d):
            out = layer(x)
            H_out, W_out = out.shape[2], out.shape[3]
            flops = (
                H_out * W_out *
                layer.in_channels *
                layer.out_channels *
                layer.kernel_size[0] * layer.kernel_size[1]
            )
            flops_per_layer[conv_idx] = flops
            conv_idx += 1
            x = out
        else:
            x = layer(x)  # Forward non-conv layers too

    # FC layers
    flops_per_layer['fc1'] = model.fc1[0].in_features * model.fc1[0].out_features
    flops_per_layer['fc2'] = model.fc2[0].in_features * model.fc2[0].out_features
    flops_per_layer['fc3'] = model.fc3.in_features * model.fc3.out_features

    return flops_per_layer


def compute_flops_per_segment(model, flops_dict, split_config, last_active_node_id):
    """
    Maps split configuration to FLOPs for each segment.
    """
    def flatten_layers(module):
        flat = []
        for layer in module.children():
            if isinstance(layer, nn.Sequential):
                flat.extend(flatten_layers(layer))
            else:
                flat.append(layer)
        return flat

    layers = flatten_layers(model.conv_layers)
    conv_indices = [i for i, l in enumerate(layers) if isinstance(l, nn.Conv2d)]
    seq_to_conv = {seq_idx: conv_idx for conv_idx, seq_idx in enumerate(conv_indices)}

    flops_per_segment = {}
    for node_id, start, end in split_config:
        if start == end:
            flops_per_segment[node_id] = 0
            continue

        conv_range = [
            conv_idx for seq_idx, conv_idx in seq_to_conv.items()
            if start <= seq_idx < end
        ]
        segment_flops = sum(flops_dict.get(conv_idx, 0) for conv_idx in conv_range)

        if node_id == last_active_node_id:
            segment_flops += (
                flops_dict['fc1'] +
                flops_dict['fc2'] +
                flops_dict['fc3']
            )

        flops_per_segment[node_id] = segment_flops

    return flops_per_segment

def compute_flops_per_block(flops_per_layer):
    """
    Computes flops per block of the VGG16 model and normalizes the values.
    Args:
        flops_per_layer (dict): the flops per layer of VGG16

    Returns:
        Flops per block as dict.
    """
    flops_per_block = {}
    block = 1
    flops_per_block[block] = (flops_per_layer[0] + flops_per_layer[1]) * 1e-9
    block = 2
    flops_per_block[block] = (flops_per_layer[2] + flops_per_layer[3]) * 1e-9
    block = 3
    flops_per_block[block] = (flops_per_layer[4] + flops_per_layer[5] + flops_per_layer[6]) * 1e-9
    block = 4
    flops_per_block[block] = (flops_per_layer[7] + flops_per_layer[8] + flops_per_layer[9]) * 1e-9
    block = 5
    flops_per_block[block] = (flops_per_layer[10] + flops_per_layer[11] + flops_per_layer[12]) * 1e-9
    block = 6
    flops_per_block[block] = (flops_per_layer['fc1'] + flops_per_layer['fc2'] + flops_per_layer['fc3']) * 1e-9
    return flops_per_block

