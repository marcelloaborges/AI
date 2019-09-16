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
    def __init__(self, compressed_features_size=64, channels=3, img_rows=240, img_cols=256, fc_units=1024):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=2, padding=1)

        self.flatten_size = 80 * 15 * 16

        self.fc11 = nn.Linear( self.flatten_size, fc_units )
        self.fc12 = nn.Linear( fc_units, compressed_features_size )

        self.fc21 = nn.Linear( self.flatten_size, fc_units ) 
        self.fc22 = nn.Linear( fc_units, compressed_features_size )        

    def forward(self, state):        
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        # Flatten
        x = x.view( -1, self.flatten_size )        

        mu = F.relu(self.fc11(x))
        mu = self.fc12(mu)

        logvar = F.relu(self.fc21(x))
        logvar = self.fc22(logvar)
        
        return mu, logvar

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class Decoder(nn.Module) :
    def __init__(self, compressed_features_size=64, channels=3, img_rows=240, img_cols=256, fc_units=1024):
        super(Decoder, self).__init__()

        self.img_rows = img_rows
        self.img_cols = img_cols

        self.flatten_size = 80 * 15 * 16
        
        self.fc1 = nn.Linear( compressed_features_size, fc_units)
        self.fc2 = nn.Linear( fc_units, self.flatten_size)

        self.conv_t1 = nn.ConvTranspose2d(in_channels=80, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=48, kernel_size=4, stride=2, padding=1) 
        self.conv_t3 = nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=4, stride=2, padding=1) 
        self.conv_t4 = nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1) 

    def forward(self, out_features):
        x = F.relu(self.fc1(out_features))
        x = self.fc2(x)

        x = x.view(-1, 80, 15, 16)

        x = self.conv_t1(x)
        x = F.relu(x)
        x = self.conv_t2(x)
        x = F.relu(x)
        x = self.conv_t3(x)
        x = F.relu(x)
        x = self.conv_t4(x)
        x = torch.sigmoid(x)

        return x
    
    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class DDQN(nn.Module):
    def __init__(self, encoded_state_size, action_size, fc1_units=128, fc2_units=64):
        super(DDQN, self).__init__()            

        self.fc1 = layer_init( nn.Linear(encoded_state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )        

        self.actions = layer_init( nn.Linear(fc2_units, action_size) )
        self.value = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, encoded_state):        
        # Actor
        x = F.relu( self.fc1(encoded_state) )
        x = F.relu( self.fc2(x) )
                        
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


class ICMTarget(nn.Module):

    def __init__(self, encoded_next_state_size, output_size):
        super(ICMTarget, self).__init__()

        self.features = nn.Linear(encoded_next_state_size, output_size)

    def forward(self, encoded_next_state):
        # S+1 
        features = self.features(encoded_next_state)

        return features

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class ICM(nn.Module):

    def __init__(self, encoded_state_size, action_size, output_size, inverse_fc1_units=32):
        super(ICM, self).__init__()

        self.features = nn.Linear(encoded_state_size, output_size)
        self.prediction = nn.Linear(output_size + action_size, output_size)

        self.inverse_fc1 = nn.Linear(output_size + output_size, inverse_fc1_units)
        self.inverse = nn.Linear(inverse_fc1_units, action_size)

    def forward(self, encoded_state, action, next_state_features):

        features = self.features( encoded_state )
        # S + A
        prediction = self.prediction( torch.cat( (features, action), dim=1) )

        # inverse model => action
        x_a = self.inverse_fc1( torch.cat( (features, next_state_features), dim=1 ) )
        x_a = self.inverse(x_a)
        inverse_action = F.softmax( x_a, dim=1 )

        return prediction, inverse_action

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
