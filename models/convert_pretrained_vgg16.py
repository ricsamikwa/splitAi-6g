import torch
import os
from vgg16_model import VGG16
from torchvision.models import vgg16_bn, VGG16_BN_Weights

# -----------------------
# 1. Initialize custom VGG16 with 1000 classes
# -----------------------
custom_model = VGG16(n_classes=1000)
custom_state = custom_model.state_dict()

# -----------------------
# 2. Load pretrained torchvision VGG16
# -----------------------
# torchvision_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)

torchvision_model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

pretrained_state = torchvision_model.state_dict()

# -----------------------
# 3. Map torchvision weights -> custom model keys
# -----------------------
mapped_state = {}

mapping = {
    # Conv layers mapping
    'features.0.weight': 'conv_layers.0.0.weight',
    'features.0.bias': 'conv_layers.0.0.bias',
    'features.1.weight': 'conv_layers.0.1.weight',
    'features.1.bias': 'conv_layers.0.1.bias',
    'features.1.running_mean': 'conv_layers.0.1.running_mean',
    'features.1.running_var': 'conv_layers.0.1.running_var',

    'features.3.weight': 'conv_layers.1.0.weight',
    'features.3.bias': 'conv_layers.1.0.bias',
    'features.4.weight': 'conv_layers.1.1.weight',
    'features.4.bias': 'conv_layers.1.1.bias',
    'features.4.running_mean': 'conv_layers.1.1.running_mean',
    'features.4.running_var': 'conv_layers.1.1.running_var',

    'features.7.weight': 'conv_layers.3.0.weight',
    'features.7.bias': 'conv_layers.3.0.bias',
    'features.8.weight': 'conv_layers.3.1.weight',
    'features.8.bias': 'conv_layers.3.1.bias',
    'features.8.running_mean': 'conv_layers.3.1.running_mean',
    'features.8.running_var': 'conv_layers.3.1.running_var',

    'features.10.weight': 'conv_layers.4.0.weight',
    'features.10.bias': 'conv_layers.4.0.bias',
    'features.11.weight': 'conv_layers.4.1.weight',
    'features.11.bias': 'conv_layers.4.1.bias',
    'features.11.running_mean': 'conv_layers.4.1.running_mean',
    'features.11.running_var': 'conv_layers.4.1.running_var',

    'features.14.weight': 'conv_layers.6.0.weight',
    'features.14.bias': 'conv_layers.6.0.bias',
    'features.15.weight': 'conv_layers.6.1.weight',
    'features.15.bias': 'conv_layers.6.1.bias',
    'features.15.running_mean': 'conv_layers.6.1.running_mean',
    'features.15.running_var': 'conv_layers.6.1.running_var',

    'features.17.weight': 'conv_layers.7.0.weight',
    'features.17.bias': 'conv_layers.7.0.bias',
    'features.18.weight': 'conv_layers.7.1.weight',
    'features.18.bias': 'conv_layers.7.1.bias',
    'features.18.running_mean': 'conv_layers.7.1.running_mean',
    'features.18.running_var': 'conv_layers.7.1.running_var',

    'features.20.weight': 'conv_layers.8.0.weight',
    'features.20.bias': 'conv_layers.8.0.bias',
    'features.21.weight': 'conv_layers.8.1.weight',
    'features.21.bias': 'conv_layers.8.1.bias',
    'features.21.running_mean': 'conv_layers.8.1.running_mean',
    'features.21.running_var': 'conv_layers.8.1.running_var',

    'features.24.weight': 'conv_layers.10.0.weight',
    'features.24.bias': 'conv_layers.10.0.bias',
    'features.25.weight': 'conv_layers.10.1.weight',
    'features.25.bias': 'conv_layers.10.1.bias',
    'features.25.running_mean': 'conv_layers.10.1.running_mean',
    'features.25.running_var': 'conv_layers.10.1.running_var',

    'features.27.weight': 'conv_layers.11.0.weight',
    'features.27.bias': 'conv_layers.11.0.bias',
    'features.28.weight': 'conv_layers.11.1.weight',
    'features.28.bias': 'conv_layers.11.1.bias',
    'features.28.running_mean': 'conv_layers.11.1.running_mean',
    'features.28.running_var': 'conv_layers.11.1.running_var',

    'features.30.weight': 'conv_layers.12.0.weight',
    'features.30.bias': 'conv_layers.12.0.bias',
    'features.31.weight': 'conv_layers.12.1.weight',
    'features.31.bias': 'conv_layers.12.1.bias',
    'features.31.running_mean': 'conv_layers.12.1.running_mean',
    'features.31.running_var': 'conv_layers.12.1.running_var',

    'features.34.weight': 'conv_layers.14.0.weight',
    'features.34.bias': 'conv_layers.14.0.bias',
    'features.35.weight': 'conv_layers.14.1.weight',
    'features.35.bias': 'conv_layers.14.1.bias',
    'features.35.running_mean': 'conv_layers.14.1.running_mean',
    'features.35.running_var': 'conv_layers.14.1.running_var',

    'features.37.weight': 'conv_layers.15.0.weight',
    'features.37.bias': 'conv_layers.15.0.bias',
    'features.38.weight': 'conv_layers.15.1.weight',
    'features.38.bias': 'conv_layers.15.1.bias',
    'features.38.running_mean': 'conv_layers.15.1.running_mean',
    'features.38.running_var': 'conv_layers.15.1.running_var',

    'features.40.weight': 'conv_layers.16.0.weight',
    'features.40.bias': 'conv_layers.16.0.bias',
    'features.41.weight': 'conv_layers.16.1.weight',
    'features.41.bias': 'conv_layers.16.1.bias',
    'features.41.running_mean': 'conv_layers.16.1.running_mean',
    'features.41.running_var': 'conv_layers.16.1.running_var',

    # FC layers
    'classifier.0.weight': 'fc1.0.weight',
    'classifier.0.bias': 'fc1.0.bias',
    'classifier.3.weight': 'fc2.0.weight',
    'classifier.3.bias': 'fc2.0.bias',
    'classifier.6.weight': 'fc3.weight',
    'classifier.6.bias': 'fc3.bias',
}

# -----------------------
# 4. Transfer weights
# -----------------------
for src_key, tgt_key in mapping.items():
    if src_key in pretrained_state and tgt_key in custom_state:
        mapped_state[tgt_key] = pretrained_state[src_key]

# Update custom model state
custom_state.update(mapped_state)
custom_model.load_state_dict(custom_state)

# -----------------------
# 5. Save converted weights
# -----------------------
save_path = os.path.join("models", "vgg16-modify.pth")
torch.save(custom_model.state_dict(), save_path)
print(f"Converted weights saved to {save_path}")
