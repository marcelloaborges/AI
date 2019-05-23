import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class CNNDDQN(nn.Module):
    def __init__(self, DEVICE, action_size, channels=3, img_rows=256, img_cols=240, 
        gru_hidden_units=512, fc1_units=256):
        super(CNNDDQN, self).__init__() 
                   
        self.DEVICE = DEVICE
            
        # CONV

        self.conv1 = nn.Conv2d( channels, 8, kernel_size=7, stride=1, padding=3 )
        self.pool1 = nn.MaxPool2d( kernel_size=2 )

        self.conv2 = nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding=2 )
        self.pool2 = nn.MaxPool2d( kernel_size=2 )

        self.conv3 = nn.Conv2d( 16, 32, kernel_size=5, stride=1, padding=2 )
        self.pool3 = nn.MaxPool2d( kernel_size=2 )

        self.conv4 = nn.Conv2d( 32, 64, kernel_size=3, stride=1, padding=1 )
        self.pool4 = nn.MaxPool2d( kernel_size=2 )

        self.state_size = 64 * 16 * 15                

        # GRU

        self.gru_hidden_units = gru_hidden_units

        self.gru_x2h = layer_init( nn.Linear( self.state_size, gru_hidden_units * 3 ) )
        self.gru_h2h = layer_init( nn.Linear( gru_hidden_units, gru_hidden_units * 3 ) )

        # FC 512x256x7(1)

        self.fc1 = layer_init( nn.Linear(gru_hidden_units, fc1_units) )

        self.fc_actions = layer_init( nn.Linear(fc1_units, action_size) )

        self.fc_value = layer_init( nn.Linear(fc1_units, 1) )

    # def init_hidden(self, batch_size):
    #     weight = next(self.lstm.parameters()).data
    #     return (weight.new(self.lstm_layers, batch_size, self.lstm_units).zero_(),
    #             weight.new(self.lstm_layers, batch_size, self.lstm_units).zero_())


    def forward(self, state, hidden):
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

        # GRU    
        gate_x = self.gru_x2h( x )
        gate_h = self.gru_h2h( hidden )

        # gate_x = gate_x.squeeze(0)
        # gate_h = gate_h.squeeze(0)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
                
        resetgate = F.sigmoid( i_r + h_r )
        inputgate = F.sigmoid( i_i + h_i )
        newgate = F.tanh( i_n + ( resetgate * h_n ) )
        
        hy = newgate + inputgate * ( hidden - newgate )
        x = hy

        # Actor
        x = F.relu( self.fc1(x) )
        
        actions = self.fc_actions(x)
        value = self.fc_value(x)
        
        return value + actions - actions.mean(), hy

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)