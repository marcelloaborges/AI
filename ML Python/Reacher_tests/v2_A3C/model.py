import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class AC_Model(nn.Module):
    
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=32):

        super(AC_Model, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)        

        self.fc_action = nn.Linear(fc2_units, action_size)

        self.fc_critic = nn.Linear(fc2_units, 1)
    
    def forward(self, state, action=None):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)

        
        a = self.fc_action(x)
        act = F.tanh(a)
        
        critic = self.fc_critic(x)

        return act, critic

        # if action:
        #     v = torch.cat((x, action), dim=1)
        #     critic = self.fc_critic(v)

        # return act, critic

