"""
agent.py

Defines the RL agent and its associated parameters to train or infer the RL algorithm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rl_utils import load_model_params
from utils.split_generator import generate_random_split
from utils.action_space import enumerate_action_space
from rl.replay_buffer import ReplayBuffer


class DDQNAgent(nn.Module):
    def __init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        self.n_states = n_states
        self.n_actions = n_actions
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.action_indices = [k for k in range(self.n_actions)]
        self.layer1 = nn.Linear(self.n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)
        self.total_flops_offloaded = 0
        self.replay_buffer = ReplayBuffer(capacity=self.scenario_params['buffer_size'])
        self.discount_factor = self.scenario_params['discount_factor']
        self.batch_size = self.scenario_params['batch_size']
        self.loss = []
        self.loss_counter = 0

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def load_model(self, episode_count, nn_type):
        if episode_count > 1:
            agent = load_model_params('ddqn', nn_type, self.scenario_params, episode_count)
            self.load_state_dict(agent)
        else:
            self.load_state_dict(torch.load('rl/initial_models/main_params_ddqn.pt'))

    def get_agent_state(self, episode_params, flops_per_block):
        state = np.zeros((self.n_states,))
        idx = 0
        state[idx] = episode_params['ue_bandwidth']
        idx += 1
        state[idx] = episode_params['ue_freq']
        idx += 1
        state[idx] = episode_params['ue_flops_cycle']
        idx += 1
        state[idx] = episode_params['bandwidth']
        idx += 1
        state[idx] = episode_params['energy_cost']
        idx += 1
        state[idx] = episode_params['power']
        idx += 1
        for node_id in range(1, self.num_nodes):
        # freq, flops per cycle
            state[idx] = episode_params['freqs'][node_id - 1]
            idx += 1
            state[idx] = episode_params['flops_cycle'][node_id - 1]
        idx += 1
        # capture flops per block
        for block in range(1, 7):
            state[idx] = flops_per_block[block]
            idx += 1
        # finally, the energy credit
        state[idx] = self.total_flops_offloaded

        state = torch.Tensor(state)
        return state

    def choose_action(self, state, epsilon):
        playable_actions = enumerate_action_space(self.allowed_splits, self.num_nodes)
        n_actions = len(playable_actions)
        random_value = np.random.random()
        # agent explores by selecting a random split config
        if epsilon > random_value and not self.scenario_params['inference']:
            split_config = generate_random_split(self.allowed_splits, self.num_nodes)
        # agent exploits the current learned knowledge by selecting the action with the highest Q-value
        elif epsilon <= random_value or self.params_config['inference']:
            with torch.no_grad():
                playable_action_indx = [k for k in range(n_actions)]
                playable_action_indx = torch.LongTensor(playable_action_indx)
                action_idx = self(state.clone().detach().float())[playable_action_indx].argmax().item()
                split_config = playable_actions[action_idx]
        return split_config


    def perform_action(self):
        # for the selected split config (or action)
        # check 1) if the energy credit budget is satisfied, 2) inference latency is below the allowed limit
        # if yes, then "perform" the split, mark it as a "successful" split, update the flops offloaded
        # if no, then ue cannot offload any layers to the network, computes everything on its own, mark it as "unsuccessful"
        pass

    def get_instant_reward(self, inference_time, ue_energy_comp, ue_energy_comm):
        # compute inference time and ue energy using the selected split config
        return 1

    def update_total_flops_offloaded(self, split_config):
        for i, (node_id, start, end) in enumerate(split_config):
            if node_id == 0:    # skip if the node is the ue
                continue
            #else:
                #if start != end:


    def check_energy_credit_budget(self, split_config):
        pass




class QValues:
    device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')

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
