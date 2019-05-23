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
    def __init__(self, DEVICE, action_size, img_rows=240, img_cols=256, rgb=True, fc1_units=512, fc2_units=512):
        super(CNNDDQN, self).__init__() 
                   
        self.DEVICE = DEVICE
            
        # CONV

        self.conv1 = torch.nn.Conv2d( 3, 8, kernel_size=7, stride=1, padding=3 )
        self.pool1 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv2 = torch.nn.Conv2d( 8, 16, kernel_size=5, stride=1, padding=2 )
        self.pool2 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv3 = torch.nn.Conv2d( 16, 32, kernel_size=5, stride=1, padding=2 )
        self.pool3 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv4 = torch.nn.Conv2d( 32, 64, kernel_size=3, stride=1, padding=1 )
        self.pool4 = torch.nn.MaxPool2d( kernel_size=2 )

        self.state_size = 64 * 16 * 15        

        # HIDDEN NAC
        mu, sigma = 0, 0.1

        sm1 = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001                
        self.m1 = torch.tensor( sm1 ).float().to(self.DEVICE)

        sw1 = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001
        self.w1 = torch.tensor( sw1 ).float().to(self.DEVICE)

        sg = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001
        self.g = torch.tensor( sg ).float().to(self.DEVICE)

        # FC 1024x512x7        

        self.fc1 = layer_init( nn.Linear(fc1_units, fc1_units) )

        self.fc_actions = layer_init( nn.Linear(fc1_units, action_size) )

        self.fc_value = layer_init( nn.Linear(fc1_units, 1) )


    def forward(self, state, action=None):
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
        x = x.view( -1, self.state_size )

        # NAC 
        # x * ( tan(w) * sig(m) )
        gc = F.sigmoid( torch.mm( x , self.g ) )
        a = torch.mm( x, ( F.tanh( self.w1 ) * F.sigmoid( self.m1 ) ) )        
        m = torch.exp( torch.mm( torch.log( torch.abs( x ) + 1e-10 ), F.tanh( self.w1 ) * F.sigmoid( self.m1 ) ) )

        x = ( gc * a ) + ( ( 1 - gc ) * m )

        # Actor
        x = F.relu( self.fc1(x) )
        
        actions = self.fc_actions(x)
        value = self.fc_value(x)
        
        return F.sigmoid( value + actions - actions.mean() )

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)