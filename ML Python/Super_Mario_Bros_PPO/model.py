import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch.optim as optim

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def kaiming_layer_init(layer, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    return layer

def kaiming_weight_init(weight, mode='fan_out', nonlinearity='relu'):
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    return weight

def orthogonal_initialize_weights(modules):
    for module in modules:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(module.weight)
            # nn.init.kaiming_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)


class PPOModel(nn.Module):
    def __init__(self, n_frames, action_size):
        super(PPOModel, self).__init__()
        self.conv1 = nn.Conv2d(n_frames, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)        
        # self.linear1 = nn.Linear(32 * 6 * 16, 1024)
        # self.linear2 = nn.Linear(1024, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, action=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view( x.size(0), -1 )
        x = self.linear(x)
        # x = self.linear1(x)        
        # x = self.linear2(x)

        # ACTOR
        logits = self.actor_linear(x)
        probs = F.softmax( logits, dim=1 )
        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        # CRITIC
        value = self.critic_linear(x)

        return action, probs, log_prob, entropy, value

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)


class ActorModel(nn.Module):

    def __init__(self, n_frames, action_size):
        super(ActorModel, self).__init__()       

        # EMBEDDING
        self.conv1 = nn.Conv2d(n_frames, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)

        # ACTION
        self.actor_linear = nn.Linear(512, action_size)
        
        orthogonal_initialize_weights(self.modules())

    def forward(self, state, action=None):

        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view( x.size(0), -1 )
        x = self.linear(x)

        # ACTOR
        logits = self.actor_linear(x)
        probs = F.softmax( logits, dim=1 )
        dist = Categorical( probs )

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()

        return action, probs, log_prob, entropy

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)

class CriticModel(nn.Module):

    def __init__(self, n_frames):
        super(CriticModel, self).__init__()         

        # EMBEDDING
        self.conv1 = nn.Conv2d(n_frames, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)

        # VALUE
        self.critic_linear = nn.Linear(512, 1)
        
        orthogonal_initialize_weights(self.modules())

    def forward(self, state):
        
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view( x.size(0), -1 )
        x = self.linear(x)

        # CRITIC
        value = self.critic_linear(x)

        return value

    def load(self, checkpoint, device:'cpu'):
        if os.path.isfile(checkpoint):
            self.load_state_dict(torch.load(checkpoint, map_location={'cuda:0': device.type}))

    def checkpoint(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)