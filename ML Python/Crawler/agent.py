import numpy as np
import random
import os

import torch
import torch.optim as optim

from model import ActorModel

class Agent:

    def __init__(self, 
        device,
        key,
        state_size,
        action_size,
        lr,
        weight_decay,
        memory,
        noise, 
        checkpoint_folder):

        self.DEVICE = device
        self.KEY = key        
        self.checkpoint_file = checkpoint_folder + 'checkpoint_actor_' + str(self.KEY) + '.pth'

        # Actor Network (w/ Target Network)
        self.actor_model = ActorModel(state_size, action_size).to(self.DEVICE)
        self.actor_target_model = ActorModel(state_size, action_size).to(self.DEVICE)
        self.actor_optim = optim.Adam(self.actor_model.parameters(), lr=lr, weight_decay=weight_decay)

        self.actor_model.load(self.checkpoint_file)
        self.actor_target_model.load(self.checkpoint_file)

        # Replay memory        
        self.memory = memory

        # Noise process
        self.noise = noise             
        
    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(state).cpu().data.numpy()
        self.actor_model.train()        

        if add_noise:
            action += self.noise.sample()
    
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward        
        self.memory.add(state, action, reward, next_state, done)

    def reset(self):
        self.noise.reset()

    def checkpoint(self):
        self.actor_model.checkpoint( self.checkpoint_file )