import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, feed_size, output_size, fc1_units, fc2_units):
        super().__init__()                
        
        self.fc1 = nn.Linear(feed_size, fc1_units)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc1_units, fc2_units)        
        self.fcOut = nn.Linear(fc2_units, output_size)
                
    def forward(self, x):        
        x = F.relu( self.fc1(x) )
        x = self.dout( x )
        x = F.relu( self.fc2(x) )
        prob = torch.sigmoid(self.fcOut(x) )        
        
        return prob
    
    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
