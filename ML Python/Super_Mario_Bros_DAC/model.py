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

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # FLATTEN IMG EMBEDDING
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        
        self.state_size = 16 * 7 * 8

        self.dropout25 = nn.Dropout(.25)
        self.dropout15 = nn.Dropout(.15)
        self.dropout10 = nn.Dropout(.1)

    def forward(self, state):        

        x = state
        dims_x = x.shape
        
        # IMG FLATTEN
        x = self.conv1( x )
        x = F.relu( x )
        x = self.dropout25( x )

        x = self.conv2( x )
        x = F.relu( x )
        x = self.dropout15( x )

        x = self.conv3( x )
        x = F.relu( x )
        x = self.dropout10( x )

        x = self.conv4( x )
        x = F.relu( x )
        x = self.dropout10( x )

        # [ BATCH, SEQ(N_ITEMS), FEATURES ]        
        x = x.view( -1, self.state_size )

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class DQNModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(DQNModel, self).__init__() 
        
        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )                

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) )        

        self.dropout = nn.Dropout(.25)

    def forward(self, state):

        # x = self.dropout(state)

        x = state
        dims_x = x.shape
                
        # DQN
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )        

        x = self.fc_action(x)

        return x

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class ActorModel(nn.Module):

    def __init__(self, action_size, fc1_units=512, fc2_units=256):
        super(ActorModel, self).__init__() 

        # FLATTEN IMG EMBEDDING
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        self.flatten_size = 16 * 12 * 15

        # ACTION
        self.fc1 = layer_init( nn.Linear(self.flatten_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) )  

        self.dropout25 = nn.Dropout(.25)
        self.dropout15 = nn.Dropout(.15)
        self.dropout10 = nn.Dropout(.1)

    def forward(self, state, action=None):

        x = self.dropout25(state)

        x = state
        dims_x = x.shape
        
        # IMG FLATTEN
        x = self.conv1( x )
        x = F.relu( x )
        x = self.dropout25( x )

        x = self.conv2( x )        
        x = F.relu( x )
        x = self.dropout15( x )

        x = self.conv3( x )
        x = F.relu( x )
        x = self.dropout10( x )

        # [ BATCH, SEQ(N_ITEMS), FEATURES ]        
        x = x.view( -1, self.flatten_size )

        # ACTION
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )

        probs = F.softmax( self.fc_action(x), dim=1 )

        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        return action, probs, log_prob, entropy

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class CriticModel(nn.Module):

    def __init__(self, fc1_units=512, fc2_units=256):
        super(CriticModel, self).__init__() 

        # FLATTEN IMG EMBEDDING
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        self.flatten_size = 16 * 12 * 15

        self.fc1 = layer_init( nn.Linear(self.flatten_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )
        
        self.fc_critic = layer_init( nn.Linear(fc2_units, 1) )

        self.dropout25 = nn.Dropout(.25)
        self.dropout15 = nn.Dropout(.15)
        self.dropout10 = nn.Dropout(.1)

    def forward(self, state):
        
        x = self.dropout25(state)

        x = state
        dims_x = x.shape
        
        # IMG FLATTEN
        x = self.conv1( x )
        x = F.relu( x )
        x = self.dropout25( x )

        x = self.conv2( x )        
        x = F.relu( x )
        x = self.dropout15( x )

        x = self.conv3( x )
        x = F.relu( x )
        x = self.dropout10( x )

        # [ BATCH, SEQ(N_ITEMS), FEATURES ]        
        x = x.view( -1, self.flatten_size )

        # CRITIC
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        
        value = self.fc_critic(x)

        return value

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class RNDTargetModel(nn.Module):

    def __init__(self, state_size, output_size=64):
        super(RNDTargetModel, self).__init__()

        self.target = layer_init( nn.Linear(state_size, output_size) )

    def forward(self, state):        
        # TARGET

        target = self.target( state )
      
        return target

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class RNDPredictorModel(nn.Module):

    def __init__(self, state_size, output_size=64, fc1_units=256):
        super(RNDPredictorModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.predictor = layer_init( nn.Linear(fc1_units, output_size) )        

    def forward(self, state):
        # PREDICTOR        
        
        x = F.relu( self.fc1( state ) )
        prediction = self.predictor( x )
        
        return prediction

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)