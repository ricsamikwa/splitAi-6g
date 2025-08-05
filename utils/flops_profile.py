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
    x = torch.randn(1, *input_size)

    layer_idx = 0
    for layer in model.conv_layers:
        if isinstance(layer, nn.Conv2d):
            out = layer(x)
            H_out, W_out = out.shape[2], out.shape[3]
            flops = (
                H_out * W_out *
                layer.in_channels *
                layer.out_channels *
                layer.kernel_size[0] * layer.kernel_size[1]
            )
            flops_per_layer[layer_idx] = flops
            x = out
            layer_idx += 1
        else:
            x = layer(x)  # MaxPool, ReLU, BN (low FLOPs)

    # Fully connected layers FLOPs
    fc_flops = {
        'fc1': model.fc1[0].in_features * model.fc1[0].out_features,
        'fc2': model.fc2[0].in_features * model.fc2[0].out_features,
        'fc3': model.fc3.in_features * model.fc3.out_features
    }
    flops_per_layer.update(fc_flops)

    return flops_per_layer
