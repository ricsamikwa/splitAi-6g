import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.action_space import enumerate_action_space, extended_action_space

allowed_splits = [0, 3, 6, 10, 14, 18]  # Safe boundaries (post-MaxPool layers)
n_nodes = 4
n_states = 33   # 21 for icc paper, 27 for production dataset
compression_rates = [1.0, 0.75, 0.50, 0.25]
#compression_rates = [1.0, 0.875, 0.75, 0.625, 0.5]
split_choices, _ = enumerate_action_space(allowed_splits, n_nodes, allow_empty_nodes=True)
full_action_space, _ = extended_action_space(split_choices, compression_rates)
n_actions = len(full_action_space)
print(full_action_space)
print(n_actions)

class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(n_states, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# initialize the model
model = Model(n_states, n_actions)

##--------------- uncomment for DDQN ----------------
# specify the file location
outfile = 'initial_models/main_params_ddqn.pt'

# save the model state dict
#torch.save(model.state_dict(), outfile)
##--------------- uncomment for DDQN ----------------

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        nn.Module.__init__(self)
        #self.layer1 = nn.Linear(n_states, 128)
        #self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, n_actions)
        self.model = nn.Sequential(nn.Linear(n_states, 256), nn.Tanh(),
                                   nn.Linear(256, 256), nn.Tanh(),
                                   nn.Linear(256, n_actions))


    def forward(self, x):
        #x = F.tanh(self.layer1(x))
        #x = F.tanh(self.layer2(x))
        x = F.softmax(self.model(x), dim=0)
        return x

class Critic(nn.Module):
    def __init__(self, n_states):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(n_states, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

actor = Actor(n_states, n_actions)
critic = Critic(n_states)
outfile_actor = 'initial_models/actor_params_a2c.pt'
outfile_critic = 'initial_models/critic_params_a2c.pt'
torch.save(actor.state_dict(), outfile_actor)
torch.save(critic.state_dict(), outfile_critic)
