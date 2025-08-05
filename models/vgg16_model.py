import torch
import torch.nn as nn

def conv_layer(ch_in, ch_out, k_size, p_size):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(ch_out),
        nn.ReLU()
    )

def vgg_conv_block(in_list, out_list, k_list, p_list, pool_k, pool_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_s))
    return layers

def vgg_fc_layer(size_in, size_out):
    return nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )

class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()
        block1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        block2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        block3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        block4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        block5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        
        self.conv_layers = nn.Sequential(*block1, *block2, *block3, *block4, *block5)
        self.fc1 = vgg_fc_layer(7*7*512, 4096)
        self.fc2 = vgg_fc_layer(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

    def forward(self, x, start_layer=0, end_layer=None):
        layers = list(self.conv_layers.children())
        if end_layer is None:
            end_layer = len(layers)
        for i in range(start_layer, end_layer):
            x = layers[i](x)
        return x
