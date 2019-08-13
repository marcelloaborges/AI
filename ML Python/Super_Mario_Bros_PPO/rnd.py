import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

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

