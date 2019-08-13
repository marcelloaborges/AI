import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class D4PGActor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(D4PGActor, self).__init__()        

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )        

        self.actions = layer_init( nn.Linear(fc2_units, action_size) )

    def forward(self, state):        
        # Actor
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
        
        actions_values = self.actions(x)
        # actions_values = torch.tanh( self.actions(x) )        
                      
        return actions_values

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class D4PGCritic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(D4PGCritic, self).__init__()        

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units + action_size, fc2_units) )

        self.values = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state, action):        
        # Actor
        x = F.relu( self.fc1(state) )

        x = torch.cat( (x, action), dim=1 )
        x = F.relu( self.fc2(x) )
        
        critic_values = self.values(x)
        # critic_values = torch.tanh( self.values(x) )                
                        
        return critic_values

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)