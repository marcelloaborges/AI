import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class ActorModel(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):        
        super(ActorModel, self).__init__()        
        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        self.fc_prob = layer_init( nn.Linear(fc2_units, action_size) )
        
        # self.bn1 = nn.BatchNorm1d(num_features=state_size)

    def forward(self, state, action=None):   
        # x = self.bn1(state)     
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        x_prob = F.softmax( self.fc_prob(x), dim=1 )        
                
        dist = Categorical(logits=x_prob)

        if action is None:            
            action = dist.sample().unsqueeze(1)
                        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()        

        return action, log_prob, entropy

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class CriticModel(nn.Module):
    def __init__(self, state_size, teammate_state_size, fc1_units=256, fc2_units=128):
        super(CriticModel, self).__init__()        
        self.fc1 = layer_init( nn.Linear(state_size + teammate_state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        self.fc_value = layer_init( nn.Linear(fc2_units, 1) )
        
        # self.bn1 = nn.BatchNorm1d(num_features=state_size)

    def forward(self, state, teammate_state):
        # x = self.bn1(state)
        s = torch.cat((state, teammate_state), dim=1)
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
                
        value = self.fc_value(x)
        
        return value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)