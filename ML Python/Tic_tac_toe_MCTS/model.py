import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Policy(nn.Module):    

    def __init__(self, state_size, action_size, fc1_units=32):
        super(Policy, self).__init__()

        # FC
        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu( x )
                        
        actions = self.actions( x )
        actions = torch.softmax( x )

        return actions

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)    