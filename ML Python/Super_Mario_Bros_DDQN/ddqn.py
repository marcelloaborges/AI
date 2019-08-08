import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=512, fc2_units=256):
        super(DDQN, self).__init__()        

        # # HIDDEN NAC
        # mu, sigma = 0, 0.1

        # sm1 = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001                
        # self.m1 = torch.tensor( sm1 ).float().to(self.DEVICE)

        # sw1 = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001
        # self.w1 = torch.tensor( sw1 ).float().to(self.DEVICE)

        # sg = np.random.normal( mu, sigma, [self.state_size, fc1_units] ) * 0.0001
        # self.g = torch.tensor( sg ).float().to(self.DEVICE)

        # FC 1024x512x7        

        self.fc1 = layer_init( nn.Linear(state_size, fc1_units) )

        self.fc_actions = layer_init( nn.Linear(fc1_units, fc2_units) )
        self.fc_value = layer_init( nn.Linear(fc1_units, fc2_units) )

        self.actions = layer_init( nn.Linear(fc2_units, action_size) )
        self.value = layer_init( nn.Linear(fc2_units, 1) )

    def forward(self, state):        
        # # NAC 
        # # x * ( tan(w) * sig(m) )
        # gc = F.sigmoid( torch.mm( x , self.g ) )
        # a = torch.mm( x, ( F.tanh( self.w1 ) * F.sigmoid( self.m1 ) ) )        
        # m = torch.exp( torch.mm( torch.log( torch.abs( x ) + 1e-10 ), F.tanh( self.w1 ) * F.sigmoid( self.m1 ) ) )

        # x = ( gc * a ) + ( ( 1 - gc ) * m )

        # Actor
        x = F.relu( self.fc1(state) )
                
        adv = F.relu( self.fc_actions(x) )
        adv = self.actions( adv )
        adv = adv - adv.mean()
        
        value = F.relu( self.fc_value(x) )
        value = self.value(value)
                
        return adv + value

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)