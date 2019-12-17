import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Encoder(nn.Module):
    def __init__(self, channels=8, compressed_features_size=32, fc1_units=256):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

        self.flatten_size = 32 * 10 * 9

        self.fc1_mu = nn.Linear( self.flatten_size, fc1_units )
        self.mu = nn.Linear( fc1_units, compressed_features_size )

        self.fc1_logvar = nn.Linear( self.flatten_size, fc1_units )
        self.logvar = nn.Linear( fc1_units, compressed_features_size )

    def forward(self, state):        
        # Encoder
        x = self.conv1( state )
        x = F.relu( x )
        x = self.conv2( x )
        x = F.relu( x )

        x = x.view( -1, self.flatten_size )        

        mu = self.fc1_mu( x )
        mu = F.relu( mu )
        mu = self.mu( mu )

        logvar = self.fc1_logvar( x )
        logvar = F.relu( logvar )
        logvar = self.logvar( logvar )
        
        return mu, logvar

    def load(self, checkpoint, device:'cpu'):     
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class Decoder(nn.Module):
    def __init__(self, channels=8, compressed_features_size=32, fc1_units=256):
        super(Decoder, self).__init__()        

        self.flatten_size = 32 * 10 * 9
        
        self.fc1 = nn.Linear( compressed_features_size, fc1_units)
        self.fc2 = nn.Linear( fc1_units, self.flatten_size)

        self.conv_t1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3,4), stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=(4,3), stride=3)

    def forward(self, out_features):
        x = self.fc1( out_features )
        x = F.relu( x )
        x = self.fc2( x )
        x = F.relu( x )

        x = x.view(-1, 32, 10, 9)
        
        x = self.conv_t1(x)
        x = F.relu(x)
        x = self.conv_t2(x)
        x = torch.sigmoid(x)

        return x
    
    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class DuelingDQN(nn.Module):

    def __init__(self, encoded_state_size, action_size, fc1_units=32):
        super(DuelingDQN, self).__init__()

        # FC
        self.fc1 = layer_init( nn.Linear(encoded_state_size, fc1_units) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )
        self.value = layer_init( nn.Linear(fc1_units, 1) )

    def forward(self, state):        
        x = self.fc1(state)
        x = F.relu( x )
                        
        adv = self.actions( x )
        adv = adv - adv.mean()
                
        value = self.value(x)

        actions_values = adv + value

        return actions_values

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class DQN(nn.Module):

    def __init__(self, encoded_state_size, action_size, fc1_units=32):
        super(DQN, self).__init__()

        # FC
        self.fc1 = layer_init( nn.Linear(encoded_state_size, fc1_units) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )

    def forward(self, state):        
        x = self.fc1(state)
        x = F.relu( x )
                        
        actions = self.actions( x )

        return actions

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)    

class Model(nn.Module):

    def __init__(self, action_size, channels=8, fc1_units=128, fc2_units=64):
        super(Model, self).__init__()
        
        self.channels = channels   
        self.filters_size = 32        

        # Dimensionality
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=self.filters_size, kernel_size=3, padding=1 )        
        self.pool1 = nn.MaxPool2d( 2, 2 )
        self.conv2 = nn.Conv2d(in_channels=self.filters_size, out_channels=self.filters_size, kernel_size=3, padding=1 )
        self.pool2 = nn.MaxPool2d( 2, 2 )
        self.conv3 = nn.Conv2d(in_channels=self.filters_size, out_channels=self.filters_size, kernel_size=3, padding=1 )
        self.pool3 = nn.MaxPool2d( 2, 2 )

        self.flatten_size = self.filters_size * 10 * 10

        # DQN
        self.fc1 = layer_init( nn.Linear(self.flatten_size, fc1_units) )        
        
        # self.actions = layer_init( nn.Linear(fc2_units, action_size) )

        self.actions = layer_init( nn.Linear(fc1_units, action_size) )
        self.value = layer_init( nn.Linear(fc1_units, 1) )

    def forward(self, state):        
        x = state

        # Encoding        
        x = self.conv1( x )
        x = F.relu( x )        
        x = self.pool1( x )
                
        x = self.conv2( x )
        x = F.relu( x )
        x = self.pool2( x )

        x = self.conv3( x )
        x = F.relu( x )
        x = self.pool3( x )

        x = x.view( -1, self.flatten_size )

        # DQN
        x = self.fc1( x )
        x = F.relu( x )

        # actions_values = self.actions( x )        

        adv = self.actions( x )
        adv = adv - adv.mean()
                
        value = self.value(x)

        actions_values = adv + value

        return actions_values

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)