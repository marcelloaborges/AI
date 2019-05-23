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
        lstm_units=512, lstm_layers=2, fc1_units=256):
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

        # LSTM

        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM( input_size=self.state_size, hidden_size=lstm_units, num_layers=lstm_layers )

        # FC 512x256x7(1)

        self.fc1 = layer_init( nn.Linear(lstm_units, fc1_units) )

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

        out, hidden = self.lstm( x, hidden )
        x = out        

        # Actor
        x = F.relu( self.fc1(x) )
        
        actions = self.fc_actions(x)
        value = self.fc_value(x)
        
        return value + actions - actions.mean(), hidden

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)