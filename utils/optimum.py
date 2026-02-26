"""
optimum.py

Class for generating the optimal split config for the given system variables and defined constraints.

"""
from utils.action_space import enumerate_action_space, extended_action_space
from utils.inference_utils import compute_inference
import torch
import torch.nn.functional as F

class Opt:
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        self.opt_split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        self.compression_rate = 1.0 # set default compression rate to 1.0
        # full default action
        self.split_compression_action = {'split': self.opt_split, 'compression': self.compression_rate}
        self.top1_accuracy_confidence = None    # set the top1 accuracy confidence to None
        self.energy_credit_consumed = 0.0  # energy credit consumed initially is 0%
        self.flops_offloaded = 0.0  # the instantaneous flops offloaded to the network
        self.total_flops_offloaded = 0  # captures the cumulative flops offloaded by the ue until now
        self.total_flops = 0  # captures total flops of all layers (static value)
        self.total_flops_on_ue = 0  # captures the cumulative flops computed on the ue until now
        for key, value in self.flops_per_block.items():
            self.total_flops += value

    def generate_optimal_split(self, time, ep, dnn_model, episode_params, output):
        # first determine the top1 accuracy confidence for the default split ONLY for the first instance
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.opt_split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # logic of optimum solution starts
        min_idx = 0 # helper variable to determine the index of the best action (split + compression)
        split_idx = 0   # index of the best split
        feasible_splits, split_indices = enumerate_action_space(self.allowed_splits, self.num_nodes, allow_empty_nodes=True)
        feasible_split_compression, action_indices_extended = extended_action_space(feasible_splits,
                                                                                    self.scenario_params['compression_rates'])

        #num_feasible_splits = len(feasible_splits)
        num_feasible_split_compression = len(feasible_split_compression)
        evaluations = [0 for _ in range(num_feasible_split_compression)]   # stores the optimization evaluation
        constraints_satisfied = [False for _ in range(num_feasible_split_compression)] # stores if constraints are satisfied
        top1_acc_confidences = [0 for _ in range(num_feasible_split_compression)]   # stores the top1 acc confidence for each action
        # for each feasible split and compression combination, compute the optimization and store it in a list
        # and check if constraints are satisfied, store the result in a list
        for i, split_compression in enumerate(feasible_split_compression):
            #print(split)
            split = split_compression['split']
            compression_rate = split_compression['compression']
            # compute the flops to be offloaded due to this selected split
            flops_to_be_offloaded, _ = self.get_flops_offloaded(split, self.allowed_splits_blocks)
            # compute the inference due to this selected split and compression
            inference_time, ue_en_comp, ue_en_comm, out = compute_inference(split, dnn_model, episode_params, output,
                                                                            compression_rate)
            top1_accuracy_conf = self.return_top1_accuracy_confidence(out)
            top1_acc_confidences[i] = top1_accuracy_conf
            optimization = ((self.scenario_params['weight_inference_time'] * inference_time) +
                            ((1 - self.scenario_params['weight_inference_time']) * (ue_en_comp + ue_en_comm))
                            - (self.scenario_params['weight_accuracy'] * top1_accuracy_conf))
            evaluations[i] = optimization
            # for each evaluated optimization, check if constraints are satisfied
            energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_to_be_offloaded)
            latency_criteria = self.check_latency_criteria(inference_time)
            accuracy_criteria = self.check_accuracy_confidence_criteria(top1_accuracy_conf)
            if energy_credit_criteria and latency_criteria and accuracy_criteria:
                constraints_satisfied[i] = True
            else:
                constraints_satisfied[i] = False
        #print(constraints_satisfied)
        #print(evaluations)
        # none of the feasible splits satisfies the constraints, continue with the previous split
        if True not in constraints_satisfied:
            # extract index of split config
            for k, v in split_indices.items():
                if v == self.opt_split:
                    split_idx = k
            return self.opt_split, self.compression_rate, split_idx
        else:
            # start with the first split + compression as the default
            best_split_compression_evaluation = evaluations[min_idx]
            best_split_compression = feasible_split_compression[min_idx]
            # for each split that satisfies constraints, check if it's the minimum
            for i, opt in enumerate(evaluations):
                if not constraints_satisfied[i]:
                    continue
                else:
                    if opt < best_split_compression_evaluation:
                        min_idx = i
                        best_split_compression_evaluation = opt
                        best_split_compression = feasible_split_compression[min_idx]
            #print('best split {}, min idx {}'.format(best_split, min_idx))
            # update the variables using the best split
            flops_offloaded, flops_on_ue = self.get_flops_offloaded(best_split_compression['split'], self.allowed_splits_blocks)
            # also update the energy credit consumed
            energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_offloaded)
            self.total_flops_on_ue += flops_on_ue
            if energy_credit_criteria:  # update these system variables only when the offloading criteria is satisfied, else previous value remains
                self.energy_credit_consumed = energy_credit_consumed
                self.total_flops_offloaded += flops_offloaded
                self.flops_offloaded = flops_offloaded
            else:
                self.flops_offloaded = 0.0
            self.opt_split = best_split_compression['split']
            self.compression_rate = best_split_compression['compression']
            # extract index of split config
            for k, v in split_indices.items():
                if v == self.opt_split:
                    split_idx = k
            # extract the top1 accuracy confidence for the best split
            self.top1_accuracy_confidence = top1_acc_confidences[min_idx]
            ## update the top1 accuracy confidence
            #self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
            return self.opt_split, self.compression_rate, split_idx, self.top1_accuracy_confidence

    def get_flops_offloaded(self, selected_split_config, allowed_splits_blocks):
        """
        Function that computes the flops offloaded for a selected split configuration.
        Args:
            selected_split_config (tuple): the selected split.
            allowed_splits_blocks (dict): dict containing the mapping of the blocks to the allowed splits.

        Returns:
            the flops offloaded to the network and the corresponding flops on the ue.
        """
        flops_on_ue = 0.0
        (node_id, start, end) = selected_split_config[0] # extract start and end layers of ue
        # case 1: all layers on ue, ue offloads nothing
        if start == 0 and end == 18:
            flops_on_ue = self.total_flops
            flops_offloaded = 0.0
            return flops_offloaded, flops_on_ue
        else:
            # case 2: at least one block on ue, ue offloads the rest
            for i, (block_id, block_start, block_end) in enumerate(allowed_splits_blocks):
                if end > block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                elif end == block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                    break
                else:
                    raise ValueError('Wrong mapping of blocks.')
            flops_offloaded = self.total_flops - flops_on_ue    # flops offloaded for this specific split config
            return flops_offloaded, flops_on_ue


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

    def check_accuracy_confidence_criteria(self, top1_acc_confidence):
        # first check if the new accuracy confidence is less than the previous one
        if top1_acc_confidence < self.top1_accuracy_confidence:
            # then check if the difference is within the desired percentage decrease
            if (self.top1_accuracy_confidence - top1_acc_confidence) <= (self.scenario_params['accuracy_decrease']/100):
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