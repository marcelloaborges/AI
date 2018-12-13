import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):        
        super(Model, self).__init__()        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_value = nn.Linear(fc2_units, action_size)

    def forward(self, state):           
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
                
        value = self.fc_value(x)
        
        return value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)