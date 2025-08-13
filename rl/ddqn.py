"""
agent.py

Defines the RL agent and its associated parameters to train or infer the RL algorithm
"""
import numpy as np
import torch
import torch.nn as nn
from utils.rl_utils import load_model_params
from utils.split_generator import generate_random_split
from utils.inference_utils import compute_inference
from utils.action_space import enumerate_action_space


class DDQNAgent(nn.Module):
    def __init__(self, scenario_params):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params


    def load_model(self, episode_count):
        if episode_count > 1:
            agent = load_model_params('ddqn', self.scenario_params, episode_count)
            self.load_state_dict(agent)
        else:
            self.load_state_dict(torch.load('rl/initial_models/main_params_ddqn.pt'))

    def get_agent_state(self):
        return 1

    def choose_action(self, state, epsilon, allowed_splits, num_nodes):
        playable_actions = enumerate_action_space(allowed_splits, num_nodes)
        n_actions = len(playable_actions)
        random_value = np.random.random()
        if epsilon > random_value and not self.scenario_params['inference']:
            split_config = generate_random_split(allowed_splits, num_nodes)
        elif epsilon <= random_value or self.params_config['inference']:
            with torch.no_grad():
                playable_action_indx = [k for k in range(n_actions)]
                playable_action_indx = torch.LongTensor(playable_action_indx)
                action_idx = self(state.clone().detach().float())[playable_action_indx].argmax().item()
                split_config = playable_actions[action_idx]
        return split_config


    def perform_action(self):
        pass

    def get_instant_reward(self):
        return 1


    def train_agent(self, epsilon):
        state = self.get_agent_state()
        action = self.choose_action(state, epsilon, self.allowed_splits, self.num_nodes)
        reward = self.get_instant_reward()
        self.perform_action()
        next_state = self.get_agent_state()

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
