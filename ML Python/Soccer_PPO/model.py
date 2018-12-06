import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Model(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):        
        super(Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_prob = nn.Linear(fc2_units, action_size)        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_prob.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x_prob = self.fc_prob(x)
        
        probs = F.softmax( x_prob, dim=1 )        

        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()        

        print(probs)

        return action, log_prob, entropy

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)