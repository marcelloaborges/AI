import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Model(nn.Module):

    def __init__(self, feed_size, output_size, fc1_units=128):
        super(Model, self).__init__() 

        self.fc1 = layer_init( nn.Linear(feed_size, fc1_units) )        
        
        self.fc_out = layer_init( nn.Linear(fc1_units, output_size) )

    def forward(self, feed):
        x = F.relu( self.fc1(feed) )        
        
        out = self.fc_out(x)        

        return out

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)