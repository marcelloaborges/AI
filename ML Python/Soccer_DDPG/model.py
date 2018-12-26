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

class ActorModel(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):        
        super(ActorModel, self).__init__()        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_prob = nn.Linear(fc2_units, action_size)
        
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_prob.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):   
        # x = self.bn1(state)     
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        x_prob = F.softmax( self.fc_prob(x) )        
                
        dist = Categorical(logits=x_prob)

        action = dist.sample()
        probs = dist.probs

        return action, probs
            
    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class CriticModel(nn.Module):
    def __init__(self, state_size, action_size, fsc1_units=256, fc2_units=128):
        super(CriticModel, self).__init__()        
        self.fcs1 = nn.Linear(state_size, fsc1_units)
        self.fc2 = nn.Linear(fsc1_units + action_size, fc2_units)
        self.fc_value = nn.Linear(fc2_units, 1)
        
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_value.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc_value(x)

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)