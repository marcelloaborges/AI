import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch.optim as optim

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def kaiming_layer_init(layer, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    return layer

def kaiming_weight_init(weight, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    return weight

class DQNModel(nn.Module):

    def __init__(self, action_size, fc1_units=256, fc2_units=128, fc3_units=64, fc4_units=32):
        super(DQNModel, self).__init__() 

        # FLATTEN IMG EMBEDDING
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        self.flatten_size = 16 * 12 * 15

        self.fc1 = layer_init( nn.Linear(self.flatten_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        self.fc3 = layer_init( nn.Linear(fc2_units, fc3_units) )
        self.fc4 = layer_init( nn.Linear(fc3_units, fc4_units) )

        self.dropout = nn.Dropout(.25)

        self.fc_action = layer_init( nn.Linear(fc4_units, action_size) )

    def forward(self, state):

        x = self.dropout(state)

        x = state
        dims_x = x.shape
        
        # IMG FLATTEN
        x = self.conv1( x )
        x = F.relu( x )
        x = self.conv2( x )
        x = F.relu( x )
        x = self.conv3( x )
        x = F.relu( x )

        # [ BATCH, SEQ(N_ITEMS), FEATURES ]        
        x = x.view( -1, self.flatten_size )

        # DQN
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.relu( self.fc3(x) )
        x = F.relu( self.fc4(x) )

        x = self.fc_action(x)

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class ActorModel(nn.Module):

    def __init__(self, encoding_size, action_size, fc1_units=128, fc2_units=64, device='cpu'):
        super(ActorModel, self).__init__() 

        self.encoding_size = encoding_size
        self.action_size = action_size

        self.fc1 = layer_init( nn.Linear(encoding_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        
        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) )
        self.fc_action.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu( x )

        x = self.fc2( x )
        x = F.relu( x )

        x = self.fc_action( x )

        # action dist
        x = ( x - x.mean() ) / x.std() + 1.0e-10

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class CriticModel(nn.Module):

    def __init__(self, encoding_size, action_size, fc1_units=128, fc2_units=64, device='cpu'):
        super(CriticModel, self).__init__() 
        
        self.encoding_size = encoding_size
        self.action_size = action_size 

        self.fc1 = layer_init( nn.Linear(encoding_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units + action_size, fc2_units) )
        
        self.fc_value = layer_init( nn.Linear(fc2_units, 1) )
        self.fc_value.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, dist):
        x = self.fc1(state)
        x = F.relu( x )

        x = torch.cat( (x, dist), dim=1 )

        x = self.fc2( x )
        x = F.relu( x )

        x = self.fc_value( x )

        # action dist
        # x = ( x - x.mean() ) / x.std() + 1.0e-10

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)