import pandas as pd
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from models.vgg16_model import VGG16
from utils.flops_profile import compute_flops_per_layer, compute_flops_per_block
from nodes.ue_node import UENode
from nodes.network_node import NetworkNode
from preprocessing.data_processing import read_trace_file
from utils.param_generator import read_params_from_file
from utils.split_generator import Baseline
from utils.scenario_generator import generate_scenario
from utils.inference_utils import compute_inference
from utils.optimum import Opt
from utils.logging_utils import write_logs
from rl.agent import Agent
from PIL import Image
import torchvision.transforms as transforms
from timeit import default_timer as timer

# -----------------------
# Setup and Parameters
# -----------------------
num_nodes = 4  # UE + 3 network nodes
energy_cost = 1  # in 1e-7 scale (J/byte) for UE communication
power = 5
agent = None
opt = None
baseline = None
num_input_files = 19    # number of input files to read data from
file_number = 1   # counter to set the file number

# Import the scenario params
scenario_params = generate_scenario()
start_episode = scenario_params['start_episode']
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

# Calculate the flops per layer and per block
flops_dict = compute_flops_per_layer(model)
flops_per_block = compute_flops_per_block(flops_dict)
#print(flops_per_block)
# -----------------------
# Possible Split Indices for VGG16
# -----------------------
allowed_splits = [0, 3, 6, 10, 14, 18]  # Safe boundaries (post-MaxPool layers)
# mapping block numbers to the start-end boundaries
allowed_splits_blocks = [(1, 0, 3), (2, 3, 6), (3, 6, 10), (4, 10, 14), (5, 14, 18)]
# -----------------------
# Generate split configuration according to desired algorithm
# -----------------------
for ep in range(start_episode, scenario_params['n_episodes'] + 1):
    start = timer()
    print('Episode {}'.format(ep))
    # ------------------------------
    # initialize logging variables
    inference_time_per_episode = []
    ue_energy_comp_per_episode = []
    ue_energy_comm_per_episode = []
    success_rate_per_episode = []
    flops_offloaded_per_episode = []
    total_flops_offloaded_per_episode = []
    total_flops_on_ue_per_episode = []
    energy_credit_consumed_per_episode = []
    split_config_per_episode = []
    top1_accuracy_per_episode = []
    top5_accuracy_per_episode = []
    # ------------------------------
    if scenario_params['split_algorithm'] == 2:
        # initialize agent
        agent = Agent(scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks)
        # update epsilon for ddqn based on episode number
        if scenario_params['rl_algorithm'] == 1:
            agent.agent.get_epsilon(ep)
    elif scenario_params['split_algorithm'] == 3:   # if optimal solution is selected
        # initialize solver
        opt = Opt(scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks)
    else:
        # initialize the baseline algorithm i.e. random/fixed split/ue only
        baseline = Baseline(scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks)
    # determine the file number for the episode
    if file_number > num_input_files:
        file_number = 1  # reset the counter to 1 if its value exceeds the number of input files
    print('File number {}'.format(file_number))
    for k in range(1, scenario_params['episode_duration'] + 1, scenario_params['time_interval']):
        #print('Time step {} in episode {}'.format(k, ep))
        # ------------------------ Read params from file ---------------------------------------------------
        # for now, the same randomly generated set of parameters are used for all episodes
        # first, read radio parameters from file based on the episode number
        path = 'input/episode_parameters/radio_parameters_moving_{}.csv'.format(file_number)
        df_radio_params = pd.read_csv(path)
        # then read other params
        df = read_params_from_file(episode=file_number, num_nodes=num_nodes)
        ue_freq = df['ue_freq'][k-1]
        ue_flops_cycle = df['ue_flops_per_cycle'][k-1]
        ue_bandwidth = df_radio_params['DL_bitrate'][k-1] / 8000    # convert kbps to megabytes/s
        freqs = []
        flops_cycle = []
        bandwidth = []
        for i in range(1, num_nodes):
            freqs.append(df['freqs{}'.format(i)][k-1])
            flops_cycle.append(df['flops_per_cycle{}'.format(i)][k-1])
            bandwidth.append(df['bandwidth{}'.format(i)][k-1])
        #ue_freq, ue_flops_cycle, ue_bandwidth, freqs, flops_cycle, bandwidth = generate_params(num_nodes)
        # --------------------------------------------------------------------------------------------------
        # Instantiate computation nodes
        ue = UENode(cpu_freq=ue_freq, flops_per_cycle=ue_flops_cycle, power=power)
        network_nodes = [NetworkNode(i, freqs[i - 1], flops_cycle[i - 1]) for i in range(1, num_nodes)]

        # -----------------------
        # Load and preprocess image
        # -----------------------
        rand_index = random.randint(1, 10)  # Randomly select an input number
        # filename = "input/input5.JPEG"  # Path to image input
        filename = f"input/input{rand_index}.JPEG"
        input_image = Image.open(filename).convert("RGB")
        # Print the selected input class (1-to-1 mapping with input number)
        #print(f"Input class: {rand_index}")
        # Apply preprocessing
        input_tensor = preprocess(input_image)
        # Add batch dimension and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_output = input_tensor.unsqueeze(0).to(device)
        model.to(device)

        # ----------------------------
        # Pack episode-specific params
        # ----------------------------
        ue_state = df_radio_params['State'][k-1]
        ue_state = 1 if ue_state == 'D' else 0
        episode_params = {'ue': ue,
                          'network_nodes': network_nodes,
                          'bandwidth': bandwidth,
                          'ue_bandwidth': ue_bandwidth,
                          'ue_freq': ue_freq,
                          'ue_flops_cycle': ue_flops_cycle,
                          'freqs': freqs,
                          'flops_cycle': flops_cycle,
                          'energy_cost': energy_cost,
                          'power': power,
                          'speed': df_radio_params['Speed'][k-1],
                          'rsrp': df_radio_params['RSRP'][k-1],
                          'rsrq': df_radio_params['RSRQ'][k-1],
                          'snr': df_radio_params['SNR'][k-1],
                          'cqi': df_radio_params['CQI'][k-1],
                          'ue_state': ue_state}

        if scenario_params['split_algorithm'] == 1:  # indicates random split
            split_config = baseline.generate_random_split(allowed_splits, num_nodes, True, model,
                                                          episode_params, current_output)
        elif scenario_params['split_algorithm'] == 2:   # rl agent
            split_config = agent.execute(k, ep, model, episode_params, current_output)  # agent determines the split every time_interval seconds
            #print('Energy credit consumed {} Split config {}'.format(agent.agent.energy_credit_consumed, split_config))
        elif scenario_params['split_algorithm'] == 3:   # optimal solution
            split_config = opt.generate_optimal_split(k, ep, model, episode_params, current_output)
            #print('Energy credit consumed {} Optimal split {}'.format(opt.energy_credit_consumed, split_config))
        elif scenario_params['split_algorithm'] == 4:   # fixed split
            split_config = baseline.fixed_split()
        else:   # ue only i.e. no split
            split_config = baseline.ue_computation_only()
        # compute inference using the generated split configuration
        total_time, ue_energy_comp, ue_energy_comm, current_output = compute_inference(split_config, model,
                                                                                       episode_params, current_output)

        #
        # -----------------------
        # Print and Store Results
        # -----------------------
        # print("=== Multi-Node Split AI Inference ===")
        # print(f"Node Frequencies (GHz): {freqs}")
        # print(f"Bandwidth (MB/s): {bandwidth}")
        # print(f"Total Inference Time: {total_time:.6f}s")
        # print(f"UE Energy (Compute): {ue_energy_comp:.6f} J")
        # print(f"UE Energy (Comm): {ue_energy_comm:.6f} J")
        # print()
        #
        inference_time_per_episode.append({'time_step': k, 'inference_time': total_time})
        ue_energy_comp_per_episode.append({'time_step': k, 'ue_energy_comp': ue_energy_comp})
        ue_energy_comm_per_episode.append({'time_step': k, 'ue_energy_comm': ue_energy_comm})
        split_config_per_episode.append({'time_step': k, 'split': split_config})
        if scenario_params['split_algorithm'] == 3: # optimum case
            energy_credit_consumed_per_episode.append({'time_step': k, 'energy_credit': opt.energy_credit_consumed})
            flops_offloaded_per_episode.append({'time_step': k, 'flops_off': opt.flops_offloaded})
        elif scenario_params['split_algorithm'] == 2: # rl case
            success_rate_per_episode.append({'time_step': k,
                                         'success_rate': (agent.agent.n_success / agent.agent.n_attempts_to_split) * 100})
            total_flops_offloaded_per_episode.append({'time_step': k, 'y_net': agent.agent.total_flops_offloaded})
            total_flops_on_ue_per_episode.append({'time_step': k, 'y_ue': agent.agent.total_flops_on_ue})
            energy_credit_consumed_per_episode.append({'time_step': k, 'energy_credit': agent.agent.energy_credit_consumed})
            flops_offloaded_per_episode.append({'time_step': k, 'flops_off': agent.agent.flops_offloaded})
        else:   # for all other baseline algorithms i.e. random/fixed split/ue only
            energy_credit_consumed_per_episode.append({'time_step': k, 'energy_credit': baseline.energy_credit_consumed})
            flops_offloaded_per_episode.append({'time_step': k, 'flops_off': baseline.flops_offloaded})
        # -----------------------
        # Final Classification Output
        # -----------------------
        with torch.no_grad():
            final_output = F.softmax(current_output, dim=1)

            top1_prob, top1_idx = torch.topk(final_output, 1)
            top5_prob, top5_idx = torch.topk(final_output, 5)

            #print(f"Top-1 Accuracy Confidence: {top1_idx.item()} (prob: {top1_prob.item():.4f})")

            # Display top-5 predictions with their probabilities
            # This provides insight into the model's confidence spread across multiple classes
            #print("Top-5 Predictions:")
            for i in range(top5_idx.size(1)):
                prob = top5_prob[0, i].item()
                idx = top5_idx[0, i].item()

            # Optional: sum of top-5 probabilities (should be ≤ 1)
            #print(f"Top-5 Accuracy Confidence: {top5_prob.sum().item():.4f}")
        top1_accuracy_per_episode.append({'time_step': k, 'top1': top1_prob.item()})
        #top5_accuracy_per_episode.append({'time_step': k, 'top5': top5_prob.item()})


    # --------------------------------------
    # Save logging variables in this episode
    # --------------------------------------
    data = {'inference_time': inference_time_per_episode, 'ue_energy_comp': ue_energy_comp_per_episode,
            'ue_energy_comm': ue_energy_comm_per_episode, 'success_rate': success_rate_per_episode,
            'y_net': total_flops_offloaded_per_episode, 'y_ue': total_flops_on_ue_per_episode,
            'energy_credit': energy_credit_consumed_per_episode, 'flops_off': flops_offloaded_per_episode,
            'split': split_config_per_episode,
            'top1': top1_accuracy_per_episode, 'top5': top5_accuracy_per_episode}
    write_logs(scenario_params, ep, data, agent)
    # --------------------------------------
    # Display variables in this episode
    # --------------------------------------
    if scenario_params['split_algorithm'] == 2:
        print('Cumulative episode reward {}'.format(agent.agent.cumulative_reward))

    end = timer()
    elapsed = end - start
    print('Elapsed wall clock time {} min'.format(elapsed/60))
    file_number = file_number + 1   # increment the file number by the episode number

