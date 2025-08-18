from utils.comm_utils import calculate_comm_time, calculate_comm_energy
from utils.flops_profile import compute_flops_per_layer, compute_flops_per_segment

def compute_inference(split_config, model, episode_params, output):
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
    #print(f"Split Config: {split_config}")
    for i, (node_id, start, end) in enumerate(split_config):

        if start == end:
            #print(f"Skipping Node {node_id} (no layers assigned).")
            continue  # Skip nodes with no layers

        is_last_active = (i == last_active_idx)

        if node_id == 0:
            output, comp_time, energy = episode_params['ue'].compute(
                model, output, start, end, flops_per_segment[node_id],
                include_fc=is_last_active
            )
            ue_energy_comp += energy
            total_time += comp_time
        else:
            node = episode_params['network_nodes'][node_id - 1]
            output, comp_time = node.compute(
                model, output, start, end, flops_per_segment[node_id],
                include_fc=is_last_active
            )

            total_time += comp_time

        # Communication to the next node (if exists)
        if i < len(split_config) - 1:
            # Calculate actual data size based on current tensor output
            data_size = output.numel() * output.element_size()  # bytes

            # Communication time based on bandwidth
            comm_time = calculate_comm_time(data_size, episode_params['bandwidth'][i])
            total_time += comm_time

            # UE communication energy calculation
            if node_id == 0 or split_config[i + 1][0] == 0:
                ue_energy_comm += calculate_comm_energy(data_size, episode_params['energy_cost'])

    # -----------------------
    # Print Results
    # -----------------------
    # print("=== Multi-Node Split AI Inference ===")
    # print(f"Node Frequencies (GHz): {episode_params['freqs']}")
    # print(f"Bandwidth (MB/s): {episode_params['bandwidth']}")
    # print(f"Total Inference Time: {total_time:.6f}s")
    # print(f"UE Energy (Compute): {ue_energy_comp:.6f} J")
    # print(f"UE Energy (Comm): {ue_energy_comm:.6f} J")

    return total_time, ue_energy_comp, ue_energy_comm, output