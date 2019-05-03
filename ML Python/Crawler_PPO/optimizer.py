import numpy as np
import random
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:        

    def __init__(self, 
        device,
        actor_critic_model, actor_critic_optimizer,         
        shared_memory,
        gamma, TAU):

        self.DEVICE = device

        # Actor Critic Network
        self.actor_critic_model = actor_critic_model
        self.actor_critic_optimizer = actor_critic_optimizer

        # Shared memory
        self.memory = shared_memory

        # Hyperparameters
        self.GAMMA = gamma
        self.TAU = TAU        
        
        self.loss = 0

    def learn(self):
        