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

class A2CModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(A2CModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units), 0.1 )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units), 0.1 )

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size), 1e-3 )
        self.fc_critic = layer_init( nn.Linear(fc2_units, 1), 1e-3 )

    def forward(self, state, action=None):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )

        logits = self.fc_action(x)
        value = self.fc_critic(x)

        dist = Categorical( logits=logits )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action ).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        return action, log_prob, entropy, value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)