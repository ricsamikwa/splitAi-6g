"""
a2c.py

Defines the RL agent running the A2C algorithm and its associated parameters to train or infer the RL algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rl_utils import load_model_params
from rl.ddqn import DDQNAgent

A2C_ACTOR_CHECKPOINT = 'a2c_actor_inference.pt'
A2C_CRITIC_CHECKPOINT = 'a2c_critic_inference.pt'

class A2CAgent(DDQNAgent, nn.Module):
    def __init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices):
        DDQNAgent.__init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices)
        nn.Module.__init__(self)
        self.actor = Actor(self.n_states, self.n_actions, self.scenario_params)
        self.critic = Critic(self.n_states, self.scenario_params)
        self.entropy = self.scenario_params['entropy']
        self.entropy_factor = self.scenario_params['entropy_factor']
        self.actor_loss = []
        self.critic_loss = []
        self.advantages = []
        self.entropies = []


    def load_model_a2c(self, episode_count):
        if not self.scenario_params['inference']:
            if episode_count > 1:
                agent = load_model_params('a2c', 'actor', self.scenario_params, episode_count - 1)
                self.actor.load_state_dict(agent)
                agent = load_model_params('a2c', 'critic', self.scenario_params, episode_count - 1)
                self.critic.load_state_dict(agent)
            else:
                self.actor.load_state_dict(torch.load('rl/initial_models/actor_params_a2c.pt'))
                self.critic.load_state_dict(torch.load('rl/initial_models/critic_params_a2c.pt'))
        else:
            self.actor.load_state_dict(torch.load('rl/inference_checkpoints/{}'.format(A2C_ACTOR_CHECKPOINT)))
            self.critic.load_state_dict(torch.load('rl/inference_checkpoints/{}'.format(A2C_CRITIC_CHECKPOINT)))


    def choose_action(self, playable_actions, state):
        #print('before forward call {}'.format(state._version))
        probs = self.actor(state)
        #print(probs.shape)
        #print('after forward call {}'.format(state._version))
        #self.dist = torch.distributions.Categorical(probs=probs)
        dist = torch.distributions.Categorical(probs=probs)
        if not self.scenario_params['inference']:
            if len(playable_actions) == 1:
                action_idx = torch.tensor(0)
            else:
                action_idx = dist.sample()
            #log_prob = self.dist.log_prob(action_idx)
            #selected_split_config = playable_actions[action_idx]
            #entropy = self.dist.entropy()
        else:
            if len(playable_actions) == 1:
                action_idx = torch.tensor(0)
            else:
                action_idx = probs.argmax()
        selected_split_compression = playable_actions[action_idx]

        return selected_split_compression, action_idx

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, scenario_params):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        #self.layer1 = nn.Linear(n_states, 128)
        #self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, n_actions)
        self.model = nn.Sequential(nn.Linear(n_states, 256), nn.Tanh(),
                                   nn.Linear(256, 256), nn.Tanh(),
                                   nn.Linear(256, n_actions),
                                   )
    def forward(self, x):
        #y1 = F.tanh(self.layer1(x))
        #y2 = F.tanh(self.layer2(y1))
        #y3 = F.softmax(self.layer3(y2), dim=0)
        y = self.model(x)
        y = F.softmax(y, dim=0)
        return y


class Critic(nn.Module):
    def __init__(self, n_states, scenario_params):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        self.layer1 = nn.Linear(n_states, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
