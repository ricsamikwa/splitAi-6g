import torch
import torch.nn.functional as F
from models.vgg16_model import VGG16
from nodes.ue_node import UENode
from nodes.network_node import NetworkNode
from utils.param_generator import generate_params
from utils.comm_utils import calculate_comm_time, calculate_comm_energy
from utils.split_generator import generate_random_split

# -----------------------
# Setup
# -----------------------
num_nodes = 4  # UE + 3 network nodes
total_layers = 21  # Adjust based on your custom VGG16
freqs, flops_cycle, bandwidth = generate_params(num_nodes)
energy_cost = 2e-6  # J/byte for UE communication

# Instantiate nodes
ue = UENode(cpu_freq=freqs[0], flops_per_cycle=flops_cycle[0], power=5)
network_nodes = [NetworkNode(i, freqs[i], flops_cycle[i]) for i in range(1, num_nodes)]

# Load model
model = VGG16(n_classes=1000)
x = torch.randn(1, 3, 224, 224)

# -----------------------
# Generate random split
# -----------------------
split_config = generate_random_split(total_layers, num_nodes)

# Placeholder FLOPs per segment (layer-wise calculation to be refined)
flops_per_segment = {i: 1e9 for i in range(num_nodes)}

# -----------------------
# Inference Execution
# -----------------------
total_time = 0.0
ue_energy_comp = 0.0
ue_energy_comm = 0.0

current_output = x

for i, (node_id, start, end) in enumerate(split_config):
    if node_id == 0:
        # UE computation
        current_output, comp_time, energy = ue.compute(
            model, current_output, start, end, flops_per_segment[node_id]
        )
        ue_energy_comp += energy
        total_time += comp_time
    else:
        node = network_nodes[node_id - 1]
        current_output, comp_time = node.compute(
            model, current_output, start, end, flops_per_segment[node_id]
        )
        total_time += comp_time

    # Communication to next node if exists
    if i < len(split_config) - 1:
        next_node_id = split_config[i + 1][0]
        data_size = flops_per_segment[node_id] / 10
        comm_time = calculate_comm_time(data_size, bandwidth[node_id, next_node_id])
        total_time += comm_time
        if node_id == 0 or next_node_id == 0:
            ue_energy_comm += calculate_comm_energy(data_size, energy_cost)

# -----------------------
# Print Results
# -----------------------
print("=== Multi-Node Split AI Inference ===")
print(f"Split Config: {split_config}")
print(f"Node Frequencies (GHz): {freqs}")
print(f"Bandwidth Matrix (MB/s):\n{bandwidth}")
print(f"Total Inference Time: {total_time:.6f}s")
print(f"UE Energy (Compute): {ue_energy_comp:.6f} J")
print(f"UE Energy (Comm): {ue_energy_comm:.6f} J")

# -----------------------
# Print Accuracy
# -----------------------
with torch.no_grad():
    final_output = F.softmax(current_output, dim=1)
    top1 = torch.topk(final_output, 1).indices.item()
    top3 = torch.topk(final_output, 3).indices.squeeze().tolist()
print(f"Top-1 Predicted Class: {top1}")
print(f"Top-3 Predicted Classes: {top3}")
