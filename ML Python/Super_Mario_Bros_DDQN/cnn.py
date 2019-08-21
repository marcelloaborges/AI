import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class CNN(nn.Module):
    def __init__(self, channels=3, img_rows=256, img_cols=240):
        super(CNN, self).__init__()

        # CONV        

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=2, padding=1)        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        self.state_size = 32 * 4 * 4

    def forward(self, state):
        # Conv features
        x = F.elu( self.conv1(state) )
        x = F.elu( self.conv2(x) )
        x = F.elu( self.conv3(x) )
        x = F.elu( self.conv4(x) )

        # Flatten
        x = x.view( -1, self.state_size )

        return x

    def load(self, checkpoint):        
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)