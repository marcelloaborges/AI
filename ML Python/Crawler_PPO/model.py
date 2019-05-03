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

class ActorCriticModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):

        super(ActorCriticModel, self).__init__()

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_actor_mean = layer_init( nn.Linear(fc2_units, action_size) )
        self.fc_actor_std = layer_init( nn.Linear(fc2_units, action_size) )
        self.fc_critic = layer_init( nn.Linear(fc2_units, 1) )

        self.std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, state, action=None):
        x = self.fc1( state )
        x = F.relu( x )
        x = self.fc2( x )
        x = F.relu( x ) 

        # Actor
        mean = torch.tanh( self.fc_actor_mean( x ) )
        std = F.softplus( self.fc_actor_std( x ) )
        dist = torch.distributions.Normal( mean, std )
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        # Critic        
        value = self.fc_critic( x )

        return action, log_prob, entropy, value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)