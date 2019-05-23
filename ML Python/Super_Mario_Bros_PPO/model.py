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

class CNNActorCriticModel(nn.Module):
    def __init__(self, action_size, img_rows=240, img_cols=256, rgb=True, fc1_units=1024, fc2_units=512):
        super(CNNActorCriticModel, self).__init__() 
                   
        # CONV

        # 3x256x240 kernel 7 pooling 2x2 filtro 8   = 8x128x120
        # 8x128x120 kernel 5 pooling 2x2 filtro 16  = 16x64x60
        # 16x64x60  kernel 5 pooling 2x2 filtro 32  = 32x32x30
        # 32x32x30  kernel 3 pooling 2x2 filtro 64  = 64x16x15
        # 64x16x15  kernel 3 pooling 2x2 filtro 128 = 128x8x7

        # self.conv1 = torch.nn.Conv2d( 3, 32, kernel_size=8 )
        # self.conv2 = torch.nn.Conv2d( 32, 64, kernel_size=4 )
        # self.conv3 = torch.nn.Conv2d( 64, 64, kernel_size=3 )

        self.conv1 = torch.nn.Conv2d( 3, 8, kernel_size=7, padding=(3,3) )
        self.pool1 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv2 = torch.nn.Conv2d( 8, 16, kernel_size=5, padding=(2,2) )
        self.pool2 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv3 = torch.nn.Conv2d( 16, 32, kernel_size=5, padding=(2,2) )
        self.pool3 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv4 = torch.nn.Conv2d( 32, 64, kernel_size=3, padding=(1,1) )
        self.pool4 = torch.nn.MaxPool2d( kernel_size=2 )

        self.conv5 = torch.nn.Conv2d( 64, 128, kernel_size=3, padding=(1,1) )
        self.pool5 = torch.nn.MaxPool2d( kernel_size=2 )

        # FC

        # FC 1024x512x7

        self.state_size = 128 * 8 * 7
        self.fc1 = layer_init( nn.Linear(self.state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) ) 

        self.fc_critic = layer_init( nn.Linear(fc2_units, 1) ) 

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

        x = F.relu( self.conv5(x) )
        x = self.pool5( x )

        # Flatten
        x = x.view( -1, self.state_size )


        # Actor Critic
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )

        probs = F.softmax( self.fc_action(x), dim=1 )

        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        value = self.fc_critic(x)

        return action, log_prob, entropy, value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
