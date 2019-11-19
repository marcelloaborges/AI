import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Model(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(Model, self).__init__()

        # FC 1024x512x7        

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )        

        self.actions = layer_init( nn.Linear(fc2_units, action_size) )
        self.value = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state):        
        # Actor
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )
                        
        adv = self.actions( x )
        adv = adv - adv.mean()
                
        value = self.value(x)

        actions_values = adv + value        

        return actions_values

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)