import torch
import torch.nn as nn
import torch.nn.functional as F

n_states = 30
n_actions = 64

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

# specify the file location
outfile = 'initial_models/main_params_ddqn.pt'

# save the model state dict
torch.save(model.state_dict(), outfile)