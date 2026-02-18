"""
ddqn.py

Defines the RL agent running the DDQN algorithm and its associated parameters to train or infer the RL algorithm
"""
import numpy as np
import random
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rl_utils import load_model_params
from utils.inference_utils import compute_inference
from rl.replay_buffer import ReplayBuffer

class DDQNAgent(nn.Module):
    def __init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        self.n_states = n_states
        self.n_actions = n_actions
        self.allowed_splits = allowed_splits
        self.flops_per_block = flops_per_block
        self.num_nodes = num_nodes
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        self.action_indices = [k for k in range(self.n_actions)]
        self.layer1 = nn.Linear(self.n_states, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, self.n_actions)
        self.split_indices = split_indices
        # default split cannot be none - set ue only computation to avoid exceptions
        self.split_config = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        # extract the index of the default split
        for k, v in self.split_indices.items():
            if v == self.split_config:
                self.split_idx = k
                break
        # set default compression rate to 0.5
        self.compression_rate = 0.5
        # full default action
        self.split_compression_action = {'split': self.split_config, 'compression': self.compression_rate}
        # set the top1 accuracy confidence to None
        self.top1_accuracy_confidence = None
        self.energy_credit_consumed = 0.0    # energy credit consumed initially is 0%
        self.flops_offloaded = 0.0    # the instantaneous flops offloaded to the network
        self.total_flops_offloaded = 0  # captures the cumulative flops offloaded by the ue until now
        self.total_flops = 0    # captures total flops of all layers (static value)
        self.total_flops_on_ue = 0  # captures the cumulative flops computed on the ue until now
        self.success = None # records the success of a selected split
        self.n_success = 0  # counts the number of successful selected splits
        self.n_attempts_to_split = 0    # counts the number of attempted splits
        # compute the total flops of all blocks
        #print('flops per block {}'.format(self.flops_per_block))
        for key, value in self.flops_per_block.items():
            self.total_flops += value
        self.replay_buffer = ReplayBuffer(capacity=self.scenario_params['buffer_size'])
        self.discount_factor = self.scenario_params['discount_factor']
        self.batch_size = self.scenario_params['batch_size']
        self.loss = []
        self.loss_counter = 0
        self.reward = []
        self.reward_counter = 0 # to compute running average of the rewards
        self.cumulative_reward = 0
        self.epsilon = None
        self.epsilon_ini = self.scenario_params['epsilon_ini']
        self.epsilon_step_percent = self.scenario_params['epsilon_step_percent']
        self.epsilon_fin = self.scenario_params['epsilon_fin']

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def load_model(self, episode_count, nn_type):
        if episode_count > 1:
            agent = load_model_params('ddqn', nn_type, self.scenario_params, episode_count - 1)
            self.load_state_dict(agent)
        else:
            self.load_state_dict(torch.load('rl/initial_models/main_params_ddqn.pt'))

    def get_agent_state(self, episode_params, flops_per_block):
        state = np.zeros((self.n_states,))
        idx = 0
        state[idx] = episode_params['ue_bandwidth']
        idx = idx + 1
        state[idx] = episode_params['ue_freq']
        idx = idx + 1
        state[idx] = episode_params['ue_flops_cycle']
        idx = idx + 1
        state[idx] = episode_params['energy_cost']
        idx = idx + 1
        state[idx] = episode_params['power']
        idx = idx + 1
        # ue mobility related params
        state[idx] = episode_params['speed']
        idx = idx + 1
        state[idx] = episode_params['rsrp']
        idx = idx + 1
        # uncomment this for production dataset
        #state[idx] = episode_params['rsrq']
        #idx += 1
        state[idx] = episode_params['snr']
        idx = idx + 1
        state[idx] = episode_params['cqi']
        idx = idx + 1
        # uncomment this for production dataset
        #state[idx] = episode_params['ue_state']
        #idx += 1
        # additional context in the ns-3 dataset
        state[idx] = episode_params['tb_size']
        idx = idx + 1
        state[idx] = episode_params['delay']
        idx = idx + 1
        state[idx] = episode_params['tbler']
        idx = idx + 1
        state[idx] = episode_params['ccqi']
        idx = idx + 1
        state[idx] = episode_params['ndi']
        idx = idx + 1
        state[idx] = episode_params['csinr']
        idx = idx + 1
        state[idx] = episode_params['cthr']
        idx = idx + 1
        state[idx] = episode_params['thr']
        idx = idx + 1
        for node_id in range(1, self.num_nodes):
        # freq, flops per cycle
            state[idx] = episode_params['bandwidth'][node_id - 1]
            idx = idx + 1
            state[idx] = episode_params['freqs'][node_id - 1]
            idx = idx + 1
            state[idx] = episode_params['flops_cycle'][node_id - 1]
            idx = idx + 1
        # capture flops per block
        for block in range(1, 7):
            state[idx] = flops_per_block[block]
            idx = idx + 1
        # finally, the energy credit consumed
        state[idx] = self.energy_credit_consumed
        state = torch.Tensor(state)
        #print(state)
        return state

    def choose_action(self, playable_actions, state):
        random_value = np.random.random()
        # agent explores by selecting a random split config
        if self.epsilon > random_value and not self.scenario_params['inference']:
            playable_action_indx = [k for k in range(self.n_actions)]
            action_idx = random.sample(playable_action_indx, k=1)[0]
            selected_split_config_compression = playable_actions[action_idx]
        # agent exploits the current learned knowledge by selecting the action with the highest Q-value
        elif self.epsilon <= random_value or self.params_config['inference']:
            with torch.no_grad():
                playable_action_indx = [k for k in range(self.n_actions)]
                playable_action_indx = torch.LongTensor(playable_action_indx)
                action_idx = self(state.clone().detach().float())[playable_action_indx].argmax().item()
                selected_split_config_compression = playable_actions[action_idx]
        #print(selected_split_config)
        return selected_split_config_compression


    def perform_action(self, selected_split_config_compression, allowed_splits_blocks, dnn_model, episode_params, output):
        # first determine the top1 accuracy confidence for the default split ONLY for the first instance
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split_config, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # extract the selected split and compression
        selected_split_config = selected_split_config_compression['split']
        selected_compression = selected_split_config_compression['compression']
        #print(self.split_config)
        #print(self.compression_rate)
        # update the logging variable
        self.n_attempts_to_split += 1
        # for the selected split config (or action)
        #print('Selected split config {}'.format(selected_split_config))
        # compute the flops to be offloaded due to this selected split
        flops_to_be_offloaded, flops_on_ue = self.get_flops_offloaded(selected_split_config, allowed_splits_blocks)
        # compute the inference due to this selected split
        inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(selected_split_config, dnn_model,
                                                                                    episode_params,
                                                                      output, selected_compression)
        # check 1) if the energy credit budget is satisfied based on the flops to be offloaded,
        # then check 2) if inference latency is below the allowed limit,
        # finally, check 3) if accuracy confidence does not decrease below accuracy_decrease %
        energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_to_be_offloaded)
        latency_criteria = self.check_latency_criteria(inference_time)
        accuracy_criteria = self.check_accuracy_confidence_criteria(expected_output)
        # if yes, then "perform" the split, mark it as a "successful" split, update the flops offloaded
        # if no, then ue cannot offload any layers to the network, computes everything on its own,
        # mark it as "unsuccessful", recompute inference time and ue energy for the fallback option
        if energy_credit_criteria and latency_criteria and accuracy_criteria:
            #print('here')
            # selected split config satisfies constraints
            self.split_config = selected_split_config
            self.compression_rate = selected_compression
            self.split_compression_action = {'split': self.split_config, 'compression': self.compression_rate}
            self.success = 1
            self.n_success += 1
            self.flops_offloaded = flops_to_be_offloaded
            self.total_flops_offloaded += flops_to_be_offloaded
            # update the total flops on ue
            self.total_flops_on_ue += flops_on_ue
            # energy credit consumed needs to be updated only when the ue offloads some layers to the network
            self.energy_credit_consumed = energy_credit_consumed
        else:
            self.success = -1   # do nothing, retain previous split
            self.flops_offloaded = 0.0    # as previous split is retained, flops offloaded is zero
            # goto fallback option for agent, ue computes everything, no layers offloaded to network
            # selected_split_config = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
            # recompute the inference due to the split config & compression rate from the previous iteration,
            # no need to update anything else
            #if self.compression_rate < selected_compression:
            #    self.compression_rate = selected_compression
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split_config, dnn_model,
                                                                          episode_params,
                                                                          output, self.compression_rate)
            # update split compression action
            self.split_compression_action = {'split': self.split_config, 'compression': self.compression_rate}
            # the flops on ue due to this selected split is the total flops
            #flops_on_ue = self.total_flops
        # # update the total flops on ue
        # self.total_flops_on_ue += flops_on_ue
        #print('flops on ue {}, total flops on ue {}'.format(flops_on_ue, self.total_flops_on_ue))
        #print('flops offloaded {} total flops offloaded {}'.format(flops_to_be_offloaded, self.total_flops_offloaded))
        # extract the index of the final (performed) split
        for k, v in self.split_indices.items():
            if v == self.split_config:
                self.split_idx = k
                break
        # update the top1 accuracy confidence
        self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        return inference_time, ue_en_comp, ue_en_comm, expected_output


    def get_instant_reward(self, inference_time, ue_energy_comp, ue_energy_comm, out):
        top1_accuracy_conf = self.return_top1_accuracy_confidence(out)
        optimization = (self.scenario_params['weight_inference_time'] * inference_time) + (
                    (1 - self.scenario_params['weight_inference_time']) * (ue_energy_comp + ue_energy_comm)) - (self.scenario_params['weight_accuracy'] * top1_accuracy_conf)
        reward_1 = 1 / optimization
        reward_2 = math.pow(2, (1 / optimization))
        # original reward
        reward = math.pow(10, (1 / optimization))
        #print('reward {}'.format(reward))
        return reward_1

    def get_flops_offloaded(self, selected_split_config, allowed_splits_blocks):
        flops_on_ue = 0.0
        (node_id, start, end) = selected_split_config[0] # extract start and end layers of ue
        # case 1: all layers on ue, ue offloads nothing
        if start == 0 and end == 18:
            flops_on_ue = self.total_flops
            flops_offloaded = 0.0
            #print('here')
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
            # self.total_flops_offloaded += flops_offloaded
            #print(selected_split_config)
            #print('flops offloaded {}'.format(flops_offloaded))
            return flops_offloaded, flops_on_ue


    def check_energy_credit_budget(self, flops_offloaded):
        energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (self.total_flops + self.total_flops_on_ue)
        #print('total flops offloaded {} energy credit consumed {}'.format(self.total_flops_offloaded, self.energy_credit_consumed))
        if energy_credit_consumed <= self.max_energy_credit / 100:
            return True, energy_credit_consumed
        else:
            return False, 0

    def check_latency_criteria(self, inference_time):
        if inference_time <= self.max_inference_latency:
            return True
        else:
            return False

    def check_accuracy_confidence_criteria(self, out):
        top1_acc_confidence = self.return_top1_accuracy_confidence(out)
        # first check if the new accuracy confidence is less than the previous one
        if top1_acc_confidence < self.top1_accuracy_confidence:
            # then check if the difference is within the desired percentage decrease
            if abs(self.top1_accuracy_confidence - top1_acc_confidence) <= self.scenario_params['accuracy_decrease']:
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

    def get_epsilon(self, episode_count):
        if self.scenario_params['inference']:
            self.epsilon = self.epsilon_fin
        else:
            # training mode
            if episode_count == 1:
                self.epsilon = self.epsilon_ini
                # clear contents of existing file
                file = 'logs/rl/ddqn/epsilon/epsilon.csv'
                f = open(file, "w")
                f.truncate()
                f.close()
            else:
                # read epsilon values from file
                file = 'logs/rl/ddqn/epsilon/epsilon.csv'
                data = []
                with open(file, 'r', newline='') as csv_file:
                    reader = csv.reader(csv_file)
                    for item in reader:
                        data.append(float(item[0]))
                len_data = len(data)
                prev_eps = data[len_data - 1]
                self.epsilon = prev_eps * (1 - (self.epsilon_step_percent / 100))
                if self.epsilon < self.epsilon_fin or self.epsilon <= 0.0:
                    self.epsilon = self.epsilon_fin
        print('Epsilon {}'.format(self.epsilon))


class QValues:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next_ddqn(policy_net, target_net, next_states):
        with torch.no_grad():
            actions = policy_net(next_states.float()).argmax(dim=1).detach()
            q = torch.squeeze(target_net(next_states.float()).gather(dim=1, index=actions.unsqueeze(-1)), dim=1)
            return q

    @staticmethod
    def get_next_dqn(target_net, next_states):
        with torch.no_grad():
            return target_net(next_states.float()).max(dim=1)[0].detach()

    @staticmethod
    def get_current_greedy_actions(policy_net, states, actions):
        greedy_actions = policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))
        return greedy_actions
