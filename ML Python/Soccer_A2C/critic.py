import numpy as np
import random

from model import ActorCriticNN

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Critic:        

    def __init__(self, 
        device,
        model,
        optimizer,        
        gamma_n, loss_v, loss_entropy, 
        random_seed):        

        self.DEVICE = device    

        # Neural model
        self.model = model
        self.optimizer = optimizer

        # Hyperparameters        
        self.GAMMA_N = gamma_n
        self.LOSS_V = loss_v
        self.LOSS_ENTROPY = loss_entropy                                        

        self.seed = random.seed(random_seed)        
        
        self.t_step = 0
        
    def learn(self, buffer):
        # Learn, if enough samples are available in buffer
        experiences = buffer.sample()
        if not experiences:
            return
            
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor critic ---------------------------- #
        # Calculate the target for the critic value
        _, _, _, next_values = self.model(next_states)
        rewards = rewards + self.GAMMA_N * next_values * (1 - dones)

        # Calculate the advantage        
        _, log_probs, entropy, values = self.model(states)
        advantage = (rewards - values).squeeze(1)
        
        # Calculate the loss
        loss_policy = - log_probs * advantage
        loss_value = self.LOSS_V * advantage**2        
        # entropy = self.LOSS_ENTROPY * torch.sum( log_probs )
        entropy = self.LOSS_ENTROPY * entropy

        loss_total = torch.mean( loss_policy + loss_value + entropy )

        # minimize loss
        self.optimizer.zero_grad()
        loss_total.backward()        
        self.optimizer.step()

        # empty the buffer
        buffer.clear()

