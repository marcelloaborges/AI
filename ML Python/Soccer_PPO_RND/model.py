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

class ActorCriticModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(ActorCriticModel, self).__init__() 

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2 = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.fc_action = layer_init( nn.Linear(fc2_units, action_size) )  

        self.fc_critic = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state, action=None):
        x = F.relu( self.fc1(state) )
        x = F.relu( self.fc2(x) )

        # ACTIONS
        probs = F.softmax( self.fc_action(x), dim=1 )

        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        # VALUES
        value = self.fc_critic(x)      

        return action, log_prob, entropy, value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class RNDTargetModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(RNDTargetModel, self).__init__() 

        self.fc1_target = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2_target = layer_init( nn.Linear(fc1_units, fc2_units) )

        # # Set target parameters as untrainable
        # for param in self.target.parameters():
        #     param.requires_grad = False
    
    def forward(self, state):
        x_target = F.relu( self.fc1_target(state) )
        x_target = F.relu( self.fc2_target(x_target) )

        target_feature = x_target

        return target_feature

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class RNDPredictorModel(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=128):
        super(RNDPredictorModel, self).__init__() 

        self.fc1_predictor = layer_init( nn.Linear(state_size, fc1_units) )
        self.fc2_predictor = layer_init( nn.Linear(fc1_units, fc2_units) )

        # # Set target parameters as untrainable
        # for param in self.target.parameters():
        #     param.requires_grad = False
    
    def forward(self, state):
        x_predictor = F.relu( self.fc1_predictor(state) )
        x_predictor = F.relu( self.fc2_predictor(x_predictor) )

        predict_feature = x_predictor

        return predict_feature

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

