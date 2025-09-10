"""
split_generator.py

Utility for sampling a random split configuration for model partitioning.
This module wraps around the full action space enumerator (action_space.py) 
to ensure consistency between random baselines and the RL environment.

Note:
    Some nodes may not be allocated any layers (start_layer == end_layer),
    allowing flexible configurations where fewer than num_nodes actively 
    compute layers.

    Node 0 (UE) is always assigned at least one layer to ensure proper
    handling of the raw input image.
"""

import numpy as np
from utils.action_space import enumerate_action_space
from utils.inference_utils import compute_inference

class Baseline():
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        self.split = None
        self.energy_credit_consumed = 0.0  # energy credit consumed initially is 0%
        self.total_flops_offloaded = 0  # captures the cumulative flops offloaded by the ue until now
        self.total_flops = 0  # captures total flops of all layers (static value)
        self.total_flops_on_ue = 0  # captures the cumulative flops computed on the ue until now
        for key, value in self.flops_per_block.items():
            self.total_flops += value

    def generate_random_split(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        """
        Args:
            allowed_splits (list): Layer indices where splitting is safe without model refactoring
                                   (e.g., [0, 3, 6, 10, 14, 18]).
            num_nodes (int): Number of computation nodes to split the model across.

        Returns:
            list: A list of tuples (node_id, start_layer, end_layer) for each node.
        """
        # Build full action space once
        actions, _ = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)

        # Sample one action uniformly
        idx = np.random.randint(len(actions))

        selected_split = actions[idx]
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(selected_split)
        # compute the inference due to this selected split
        inference_time, ue_en_comp, ue_en_comm, _ = compute_inference(selected_split, dnn_model, episode_params, output)
        energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_offloaded)
        latency_criteria = self.check_latency_criteria(inference_time)
        # if both criteria are satisfied, then selected_split is the final split, else do nothing
        if energy_credit_criteria and latency_criteria:
            self.split = selected_split
            # update the flops offloaded, flops on ue and energy credit usage
            self.update_energy_credit_usage(flops_offloaded, flops_on_ue)

        return self.split

    def fixed_split(self):
        self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        return self.split

    def ue_computation_only(self):
        self.split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        return self.split

    def update_energy_credit_usage(self, flops_offloaded, flops_on_ue):
        # update the energy credit usage
        self.energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (
                self.total_flops + self.total_flops_on_ue)
        self.total_flops_offloaded += flops_offloaded
        self.total_flops_on_ue += flops_on_ue

    def check_energy_credit_budget(self, flops_offloaded):
        """
        Function that checks the energy credit criteria.
        Args:
            flops_offloaded (float): the flops offloaded due to the current split.

        Returns:
            The result of the constraint being satisfied or not, and the energy credit consumed.
        """
        energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (self.total_flops + self.total_flops_on_ue)
        if energy_credit_consumed <= self.max_energy_credit / 100:
            return True, energy_credit_consumed
        else:
            return False, 0

    def check_latency_criteria(self, inference_time):
        """
        Function that checks latency criteria.
        Args:
            inference_time (float): The inference time due to the current split.

        Returns:
            The result of the constraint being satisfied or not.
        """
        if inference_time <= self.max_inference_latency:
            return True
        else:
            return False

    def get_flops_offloaded(self, selected_split_config):
        """
        Function that computes the flops offloaded for a selected split configuration.
        Args:
            selected_split_config (tuple): the selected split.

        Returns:
            the flops offloaded to the network and the corresponding flops on the ue.
        """
        flops_on_ue = 0
        (node_id, start, end) = selected_split_config[0]  # extract start and end layers of ue
        # case 1: all layers on ue, ue offloads nothing
        if start == 0 and end == 18:
            flops_on_ue = self.total_flops
            flops_offloaded = 0
            return flops_offloaded, flops_on_ue
        else:
            # case 2: at least one block on ue, ue offloads the rest
            for i, (block_id, block_start, block_end) in enumerate(self.allowed_splits_blocks):
                if end > block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                elif end == block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                    break
                else:
                    raise ValueError('Wrong mapping of blocks.')
            flops_offloaded = self.total_flops - flops_on_ue  # flops offloaded for this specific split config
            return flops_offloaded, flops_on_ue


        # # Randomly pick internal split points
        # allowed_splits = sorted(allowed_splits)
        # K = len(allowed_splits)
        # assert K >= 2, "Need at least start/end boundaries"

        # # >>> FIX: allow choosing K-1 (final boundary) when empty nodes are allowed
        # if allow_empty_nodes:
        #     cut_idx_space = np.arange(1, K)      # 1..K-1  (enables 'all on UE')
        # else:
        #     cut_idx_space = np.arange(1, K - 1)  # 1..K-2  (classic internal cuts only)

        # # Choose cut indices
        # if cut_idx_space.size == 0:
        #     points_idx = np.array([0, K - 1])
        # else:
        #     if allow_empty_nodes:
        #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=True)
        #     else:
        #         if num_nodes - 1 > cut_idx_space.size:
        #             raise ValueError("Not enough unique split points for all nodes.")
        #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=False)
        #     cuts_idx = np.sort(cuts_idx)
        #     points_idx = np.concatenate(([0], cuts_idx, [K - 1]))

        # # Ensure UE (first segment) has at least one layer
        # if points_idx[1] == points_idx[0]:
        #     points_idx[1] = min(points_idx[0] + 1, K - 1)

        # # Ensure non-decreasing sequence
        # for i in range(2, len(points_idx)):
        #     if points_idx[i] < points_idx[i - 1]:
        #         points_idx[i] = points_idx[i - 1]

        # # Build splits
        # splits = []
        # for node_id in range(num_nodes):
        #     s_idx = points_idx[node_id]
        #     e_idx = points_idx[node_id + 1]
        #     start = int(allowed_splits[s_idx])
        #     end = int(allowed_splits[e_idx])
        #     splits.append((node_id, start, end))

        # return splits
    

# if __name__ == "__main__":
#     allowed_splits = [0, 3, 6, 10, 14, 18]
#     num_nodes = 4

#     print("Testing random split sampling...")
#     for _ in range(5):
#         split = generate_random_split(allowed_splits, num_nodes, allow_empty_nodes=True)
#         print(split)
    