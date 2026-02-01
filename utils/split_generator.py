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
import torch
import torch.nn.functional as F
from utils.action_space import enumerate_action_space, extended_action_space
from utils.inference_utils import compute_inference

class Baseline:
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        if self.scenario_params['split_algorithm'] == 4:    # in case of a fixed split
            self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        else:
            self.split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        self.compression_rate = 1.0  # set default compression rate to 1.0
        # full default action
        self.split_compression_action = {'split': self.split, 'compression': self.compression_rate}
        self.top1_accuracy_confidence = None  # set the top1 accuracy confidence to None
        self.flops_offloaded = 0.0  # the instantaneous flops offloaded to the network
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
        split_idx = None
        # first determine the top1 accuracy confidence for the default split ONLY for the first instance
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # Build full action space once
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        feasible_split_compression, action_indices_extended = extended_action_space(feasible_splits,
                                                                                    self.scenario_params[
                                                                                        'compression_rates'])
        # Sample one action (split + compression) uniformly
        idx = np.random.randint(len(feasible_split_compression))

        selected_split_compression = feasible_split_compression[idx]
        # compute the flops to be offloaded due to this selected split
        selected_split = selected_split_compression['split']
        selected_compression = selected_split_compression['compression']
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(selected_split)
        # compute the inference due to this selected split
        inference_time, ue_en_comp, ue_en_comm, out = compute_inference(selected_split, dnn_model, episode_params,
                                                                        output, selected_compression)
        energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_offloaded)
        latency_criteria = self.check_latency_criteria(inference_time)
        accuracy_criteria = self.check_accuracy_confidence_criteria(out)
        # if both criteria are satisfied, then selected_split is the final split, else do nothing
        if energy_credit_criteria and latency_criteria and accuracy_criteria:
            self.split = selected_split
            self.compression_rate = selected_compression
            # update the flops offloaded, flops on ue and energy credit usage
            self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        else:
            self.flops_offloaded = 0.0
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k

        return self.split, self.compression_rate, split_idx

    def fixed_split(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        split_idx = None
        self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        self.compression_rate = 0.5
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            # since the action remains the same (i.e. default value), updating this once is sufficient
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        return self.split, self.compression_rate, split_idx

    def ue_computation_only(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        split_idx = None
        self.split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        self.compression_rate = 0.5
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            # since the action remains the same (i.e. default value), updating this once is sufficient
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        return self.split, self.compression_rate, split_idx

    def update_energy_credit_usage(self, flops_offloaded, flops_on_ue):
        # update the energy credit usage
        self.energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (
                self.total_flops + self.total_flops_on_ue)
        self.total_flops_offloaded += flops_offloaded
        self.total_flops_on_ue += flops_on_ue
        self.flops_offloaded = flops_offloaded

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

    def check_accuracy_confidence_criteria(self, out):
        top1_acc_confidence = self.return_top1_accuracy_confidence(out)
        # first check if the new accuracy confidence is less than the previous one
        if top1_acc_confidence < self.top1_accuracy_confidence:
            # then check if the difference is within the desired percentage decrease
            if self.top1_accuracy_confidence - top1_acc_confidence <= self.scenario_params['accuracy_decrease']:
                return True
            else:
                return False
        else:
            return True # new accuracy confidence is greater than the previous one

    def return_top1_accuracy_confidence(self, out):
        with torch.no_grad():
            final_output = F.softmax(out, dim=1)
            top1_prob, top1_idx = torch.topk(final_output, 1)
        top1_accuracy_confidence = top1_prob.item()
        return top1_accuracy_confidence

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
    