import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from models.vgg16_model import VGG16
from utils.flops_profile import compute_flops_per_layer, compute_flops_per_segment
from nodes.ue_node import UENode
from nodes.network_node import NetworkNode
from utils.param_generator import generate_params
from utils.comm_utils import calculate_comm_time, calculate_comm_energy
from utils.split_generator import generate_random_split
from utils.scenario_generator import generate_scenario
from utils.inference_utils import compute_inference
from utils.logging_utils import write_logs
from rl.agent import Agent
from PIL import Image
import torchvision.transforms as transforms

# -----------------------
# Setup and Parameters
# -----------------------
num_nodes = 4  # UE + 3 network nodes
energy_cost = 1e-7  # J/byte for UE communication
agent = None

# Import the scenario params
scenario_params = generate_scenario()

# -----------------------
# Load Model
# -----------------------
model = VGG16()
model_dict = model.state_dict()
model_dict.update(torch.load("models/vgg16-modify.pth"))
model.load_state_dict(model_dict)
model.eval()
total_layers = len(list(model.conv_layers.children()))

# Transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize shorter side to 256
    transforms.CenterCrop(224),            # Crop to 224x224
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize(                  # Normalize for VGG
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------
# Possible Split Indices for VGG16
# -----------------------
allowed_splits = [0, 3, 6, 10, 14, 18]  # Safe boundaries (post-MaxPool layers)

# Generate random split configuration
# split_config = generate_random_split(allowed_splits, num_nodes) # Replace with RL method
# -----------------------
# Generate split configuration according to desired algorithm
# -----------------------
if scenario_params['split_algorithm'] == 2:
    agent = Agent(scenario_params, allowed_splits, num_nodes)
for ep in range(1, scenario_params['n_episodes'] + 1):
    # ------------------------------
    # initialize logging variables
    inference_time_per_episode = []
    ue_energy_comp_per_episode = []
    ue_energy_comm_per_episode = []
    # ------------------------------
    for k in range(1, scenario_params['episode_duration'], scenario_params['time_interval']):
        print('Time step {} in episode {}'.format(k, ep))
        ue_freq, ue_flops_cycle, ue_bandwidth, freqs, flops_cycle, bandwidth = generate_params(num_nodes)
        # Instantiate computation nodes
        ue = UENode(cpu_freq=ue_freq, flops_per_cycle=ue_flops_cycle, power=5)
        network_nodes = [NetworkNode(i, freqs[i - 1], flops_cycle[i - 1]) for i in range(1, num_nodes)]

        # -----------------------
        # Load and preprocess image
        # -----------------------
        rand_index = random.randint(1, 10)  # Randomly select an input number
        # filename = "input/input5.JPEG"  # Path to image input
        filename = f"input/input{rand_index}.JPEG"
        input_image = Image.open(filename).convert("RGB")
        # Print the selected input class (1-to-1 mapping with input number)
        print(f"Input class: {rand_index}")
        # Apply preprocessing
        input_tensor = preprocess(input_image)
        # Add batch dimension and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_output = input_tensor.unsqueeze(0).to(device)
        model.to(device)

        # ----------------------------
        # Pack episode-specific params
        # ----------------------------
        episode_params = {'ue': ue,
                          'network_nodes': network_nodes,
                          'current_output': current_output,
                          'bandwidth': bandwidth,
                          'freqs': freqs,
                          'energy_cost': energy_cost}

        if scenario_params['split_algorithm'] == 1:  # indicates random split
            split_config = generate_random_split(allowed_splits, num_nodes)  # Replace with RL method
        else:
            split_config = agent.execute(ep)  # agent determines the split every time_interval seconds

        # compute inference using the generated split configuration
        #compute_inference(split_config, model, episode_params)

        # Identify the index of the last node in the split configuration that was actually assigned layers.
        # This ensures we know where the active computation chain ends.
        # We then adjust the last active node so it always includes the model’s final layers,
        # guaranteeing that the output passes through all necessary layers before classification.
        last_active_idx = max(i for i, (_, s, e) in enumerate(split_config) if s != e)
        last_active_node_id = split_config[last_active_idx][0]

        flops_dict = compute_flops_per_layer(model)
        flops_per_segment = compute_flops_per_segment(model, flops_dict, split_config, last_active_node_id)

        # -----------------------
        # Inference Execution
        # -----------------------
        total_time = 0.0
        ue_energy_comp = 0.0
        ue_energy_comm = 0.0

        # Last active node to end at the final conv layer (18)
        split_config[last_active_idx] = (
            last_active_node_id,
            split_config[last_active_idx][1],
            18
        )
        print(f"Split Config: {split_config}")
        for i, (node_id, start, end) in enumerate(split_config):

            if start == end:
                print(f"Skipping Node {node_id} (no layers assigned).")
                continue  # Skip nodes with no layers

            is_last_active = (i == last_active_idx)

            if node_id == 0:
                current_output, comp_time, energy = ue.compute(
                    model, current_output, start, end, flops_per_segment[node_id],
                    include_fc=is_last_active
                )
                ue_energy_comp += energy
                total_time += comp_time
            else:
                node = network_nodes[node_id - 1]
                current_output, comp_time = node.compute(
                    model, current_output, start, end, flops_per_segment[node_id],
                    include_fc=is_last_active
                )

                total_time += comp_time


            # Communication to the next node (if exists)
            if i < len(split_config) - 1:
                # Calculate actual data size based on current tensor output
                data_size = current_output.numel() * current_output.element_size()  # bytes

                # Communication time based on bandwidth
                comm_time = calculate_comm_time(data_size, bandwidth[i])
                total_time += comm_time

                # UE communication energy calculation
                if node_id == 0 or split_config[i + 1][0] == 0:
                    ue_energy_comm += calculate_comm_energy(data_size, energy_cost)

        # -----------------------
        # Print and Store Results
        # -----------------------
        print("=== Multi-Node Split AI Inference ===")
        print(f"Node Frequencies (GHz): {freqs}")
        print(f"Bandwidth (MB/s): {bandwidth}")
        print(f"Total Inference Time: {total_time:.6f}s")
        print(f"UE Energy (Compute): {ue_energy_comp:.6f} J")
        print(f"UE Energy (Comm): {ue_energy_comm:.6f} J")

        inference_time_per_episode.append({'time_step': k, 'inference_time': total_time})
        ue_energy_comp_per_episode.append({'time_step': k, 'ue_energy_comp': ue_energy_comp})
        ue_energy_comm_per_episode.append({'time_step': k, 'ue_energy_comm': ue_energy_comm})
        # -----------------------
        # Final Classification Output
        # -----------------------
        with torch.no_grad():
            final_output = F.softmax(current_output, dim=1)

            top1_prob, top1_idx = torch.topk(final_output, 1)
            top5_prob, top5_idx = torch.topk(final_output, 5)

            print(f"Top-1 Accuracy Confidence: {top1_idx.item()} (prob: {top1_prob.item():.4f})")

            # Display top-5 predictions with their probabilities
            # This provides insight into the model's confidence spread across multiple classes
            print("Top-5 Predictions:")
            for i in range(top5_idx.size(1)):
                prob = top5_prob[0, i].item()
                idx = top5_idx[0, i].item()

            # Optional: sum of top-5 probabilities (should be ≤ 1)
            print(f"Top-5 Accuracy Confidence: {top5_prob.sum().item():.4f}")

    # --------------------------------------
    # Save logging variables in this episode
    # --------------------------------------
    write_logs(scenario_params, ep, 'inference_time', inference_time_per_episode)
    write_logs(scenario_params, ep, 'ue_energy_comp', ue_energy_comp_per_episode)
    write_logs(scenario_params, ep, 'ue_energy_comm', ue_energy_comm_per_episode)

