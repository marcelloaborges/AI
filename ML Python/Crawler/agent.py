import numpy as np
import random
import os

import torch

class Agent:

    def __init__(self, 
        device,        
        model,
        memory, noise):

        self.DEVICE = device

        # Actor Network (w/ Target Network)
        self.model = model

        # Replay memory        
        self.memory = memory

        # Noise process
        self.noise = noise        
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():
            action = self.model(state).cpu().data.numpy()
        self.model.train()        

        if add_noise:
            action += self.noise.sample()
    
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(len(state)):
            self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])

    def reset(self):
        self.noise.reset()