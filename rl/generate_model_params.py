import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.action_space import enumerate_action_space

allowed_splits = [0, 3, 6, 10, 14, 18]  # Safe boundaries (post-MaxPool layers)
n_nodes = 4
n_states = 27   # 21 for icc paper
actions, action_indices = enumerate_action_space(allowed_splits, n_nodes, allow_empty_nodes=True)
n_actions = len(actions)
print(actions)
print(n_actions)

class Model(nn.Module):
    def __init__(self, n_states, n_actions):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# initialize the model
model = Model(n_states, n_actions)

##--------------- uncomment for DDQN ----------------
# specify the file location
#outfile = 'initial_models/main_params_ddqn.pt'

# save the model state dict
#torch.save(model.state_dict(), outfile)
##--------------- uncomment for DDQN ----------------

class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        nn.Module.__init__(self)
        #self.layer1 = nn.Linear(n_states, 128)
        #self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, n_actions)
        self.model = nn.Sequential(nn.Linear(n_states, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh(),
                                   nn.Linear(128, n_actions))


    def forward(self, x):
        #x = F.tanh(self.layer1(x))
        #x = F.tanh(self.layer2(x))
        x = F.softmax(self.model(x), dim=0)
        return x

class Critic(nn.Module):
    def __init__(self, n_states):
        nn.Module.__init__(self)
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

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
