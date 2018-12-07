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

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

class Model(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):        
        super(Model, self).__init__()        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_prob = nn.Linear(fc2_units, action_size)
        self.fc_value = nn.Linear(fc2_units, 1)
        
        init_weights(self.fc1)
        init_weights(self.fc2)
        init_weights(self.fc_prob)
        init_weights(self.fc_value)

    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc_prob.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc_value.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x_prob = self.fc_prob(x)
        x_value = self.fc_value(x)
                
        dist = Categorical(logits=x_prob)

        action = dist.sample()

        prob_selection = action.unsqueeze(1)
        prob = dist.probs.gather(1, prob_selection)
        print('\rProb: \t{}'.format(prob), end="")  

        entropy = dist.entropy()

        value = x_value

        return action, prob, entropy, value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)