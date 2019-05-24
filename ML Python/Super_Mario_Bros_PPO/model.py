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

class CNNLSTMActorCriticModel(nn.Module):
    def __init__(self, action_size, channels=3, img_rows=256, img_cols=240, 
        lstm_hidden_units=512, fc1_units=256):
        super(CNNLSTMActorCriticModel, self).__init__() 
                   
        # CONV
        self.conv1 = nn.Conv2d( channels, 32, kernel_size=7, stride=1, padding=3 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )

        self.conv2 = nn.Conv2d( 32, 32, kernel_size=5, stride=1, padding=2 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )

        self.conv3 = nn.Conv2d( 32, 32, kernel_size=5, stride=1, padding=2 )
        self.pool3 = nn.MaxPool2d( kernel_size=2 )

        self.conv4 = nn.Conv2d( 32, 32, kernel_size=3, stride=1, padding=1 )
        self.pool4 = nn.MaxPool2d( kernel_size=2 )

        self.state_size = 32 * 16 * 15                

        # LSTM
        self.lstm_hidden_units = lstm_hidden_units

        self.lstm_x2h = layer_init( nn.Linear( self.state_size, lstm_hidden_units * 4 ) )
        self.lstm_h2h = layer_init( nn.Linear( lstm_hidden_units, lstm_hidden_units * 4 ) )

        # FC        
        self.fc1 = layer_init( nn.Linear(lstm_hidden_units, fc1_units) )

        self.fc_action = layer_init( nn.Linear(fc1_units, action_size) ) 

        self.fc_critic = layer_init( nn.Linear(fc1_units, 1) ) 

    def forward(self, state, hx, cx, action=None):
        # Conv features
        x = F.relu( self.conv1(state) )
        x = self.pool1( x )

        x = F.relu( self.conv2(x) )
        x = self.pool2( x )

        x = F.relu( self.conv3(x) )
        x = self.pool3( x )

        x = F.relu( self.conv4(x) )
        x = self.pool4( x )

        # Flatten
        x = x.view( -1, 1, self.state_size )

        # LSTM
        gates = self.lstm_x2h(x) + self.lstm_h2h(hx)
            
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 2)
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)        

        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, F.tanh(cy))
        
        x = hy

        # Actor Critic
        x = F.relu( self.fc1(x) )        

        probs = F.softmax( self.fc_action(x), dim=1 )

        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        value = self.fc_critic(x)

        return action, log_prob, entropy, value, hy, cy

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
