import numpy as np

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
    
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class ActorModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
                
        super(ActorModel, self).__init__()        
        
        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_out = layer_init( nn.Linear(fc2_units, action_size) )

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""        
        x = F.relu(self.fc1(state))           
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc_out(x))

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class CriticModel(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fcs1_units=512, fc2_units=256):
        
        super(CriticModel, self).__init__()        

        self.fcs1 = layer_init( nn.Linear(state_size, fcs1_units) )
        self.fc2 = layer_init( nn.Linear(fcs1_units + action_size, fc2_units) )

        self.fc_out = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""        
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)        
        x = F.relu(self.fc2(x))        

        return self.fc_out(x)

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
 