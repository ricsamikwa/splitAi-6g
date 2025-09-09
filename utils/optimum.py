"""
optimum.py

Class for generating the optimal split config for the given system variables and defined constraints.

"""
from utils.action_space import enumerate_action_space
from utils.inference_utils import compute_inference

class Opt:
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        self.opt_split = None
        self.energy_credit_consumed = 0.0  # energy credit consumed initially is 0%
        self.total_flops_offloaded = 0  # captures the cumulative flops offloaded by the ue until now
        self.total_flops = 0  # captures total flops of all layers (static value)
        self.total_flops_on_ue = 0  # captures the cumulative flops computed on the ue until now
        for key, value in self.flops_per_block.items():
            self.total_flops += value

    def generate_optimal_split(self, time, ep, dnn_model, episode_params, output):
        min_idx = 0
        feasible_splits, action_indices = enumerate_action_space(self.allowed_splits, self.num_nodes, allow_empty_nodes=True)
        num_feasible_splits = len(feasible_splits)
        evaluations = [0 for _ in range(num_feasible_splits)]   # stores the optimization evaluation
        constraints_satisfied = [False for _ in range(num_feasible_splits)] # stores if constraints are satisfied
        # for each feasible split, compute the optimization and store it in a list
        # and check if constraints are satisfied, store the result in a list
        for i, split in enumerate(feasible_splits):
            #print(split)
            # compute the flops to be offloaded due to this selected split
            flops_to_be_offloaded, flops_on_ue = self.get_flops_offloaded(split, self.allowed_splits_blocks)
            # compute the inference due to this selected split
            inference_time, ue_en_comp, ue_en_comm, _ = compute_inference(split, dnn_model, episode_params, output)
            optimization = (self.scenario_params['weight_inference_time'] * inference_time) + ((1 - self.scenario_params['weight_inference_time']) * (ue_en_comp + ue_en_comm))
            evaluations[i] = optimization
            # for each evaluated optimization, check if constraints are satisfied
            energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_to_be_offloaded)
            latency_criteria = self.check_latency_criteria(inference_time)
            if energy_credit_criteria and latency_criteria:
                constraints_satisfied[i] = True
            else:
                constraints_satisfied[i] = False
        #print(constraints_satisfied)
        #print(evaluations)
        # none of the feasible splits satisfies the constraints, continue with the previous split
        if True not in constraints_satisfied:
            return self.opt_split
        else:
            # start with the first split as the default
            best_split_evaluation = evaluations[min_idx]
            best_split = feasible_splits[min_idx]
            # for each split that satisfies constraints, check if it's the minimum
            for i, opt in enumerate(evaluations):
                if not constraints_satisfied[i]:
                    continue
                else:
                    if opt < best_split_evaluation:
                        min_idx = i
                        best_split_evaluation = opt
                        best_split = feasible_splits[min_idx]
            #print('best split {}, min idx {}'.format(best_split, min_idx))
            # update the variables using the best split
            flops_offloaded, flops_on_ue = self.get_flops_offloaded(best_split, self.allowed_splits_blocks)
            # also update the energy credit consumed
            energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_offloaded)
            self.total_flops_on_ue += flops_on_ue
            if energy_credit_criteria:  # update only when the criteria is satisfied, else previous value remains
                self.energy_credit_consumed = energy_credit_consumed
                self.total_flops_offloaded += flops_offloaded
            # however, if none of the feasible splits satisfies the constraints, go to fallback option
            #if True not in constraints_satisfied:
            #    best_split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
            #    print('No feasible solution found - ue computes everything')
            #    # the flops on ue due to this selected split is the total flops
            #    flops_on_ue = self.total_flops
            #    self.total_flops_on_ue += flops_on_ue
            #print(self.total_flops_offloaded)
            #print(self.total_flops_on_ue)
            #self.opt_split = best_split
            self.opt_split = best_split
            return best_split

    def get_flops_offloaded(self, selected_split_config, allowed_splits_blocks):
        """
        Function that computes the flops offloaded for a selected split configuration.
        Args:
            selected_split_config (tuple): the selected split.
            allowed_splits_blocks (dict): dict containing the mapping of the blocks to the allowed splits.

        Returns:
            the flops offloaded to the network and the corresponding flops on the ue.
        """
        flops_on_ue = 0
        (node_id, start, end) = selected_split_config[0] # extract start and end layers of ue
        # case 1: all layers on ue, ue offloads nothing
        if start == 0 and end == 18:
            flops_on_ue = self.total_flops
            flops_offloaded = 0
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