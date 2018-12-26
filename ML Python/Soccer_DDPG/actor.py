import numpy as np
import random
import os

from model import ActorModel

import torch
import torch.nn.functional as F
import torch.optim as optim

class Actor:

    def __init__(self, 
        device,
        key,
        local_model, target_model, optimizer,
        memory, noise,
        checkpoint_folder = './'):   

        self.DEVICE = device

        self.KEY = key

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network (w/ Target Network)
        self.local = local_model
        self.target = target_model
        self.optimizer = optimizer

        if os.path.isfile(self.CHECKPOINT_FOLDER):
            self.local.load(self.CHECKPOINT_FOLDER)
            self.target.load(self.CHECKPOINT_FOLDER)

        # Replay memory        
        self.memory = memory

        # Noise process
        self.noise = noise

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.local.eval()
        with torch.no_grad():
            action, prob = self.local(state)
        self.local.train()        

        action = action.cpu().data.numpy()
        prob = prob.cpu().data.numpy()

        if add_noise:
            prob += self.noise.sample()
            prob = np.clip(prob, 0, 1)                
        
        return action, prob

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

    def reset(self):
        self.noise.reset()
     
    def checkpoint(self):
        self.local.checkpoint(self.CHECKPOINT_FOLDER)