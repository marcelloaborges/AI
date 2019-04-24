import numpy as np
import random
import os

import torch
import torch.optim as optim

from model import ActorModel

class Agent:

    def __init__(self, 
        device,        
        actor_model,
        memory,
        noise):

        self.DEVICE = device

        # Actor Network
        self.actor_model = actor_model

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

    def step(self, keys, state, action, reward, next_state, done):
        # Save experience / reward        
        self.memory.add( 
            np.array( keys ).reshape( -1, 1 ),
            state,
            action,
            np.array( reward ).reshape( -1, 1 ),
            next_state,
            np.array( done ).reshape( -1, 1 )
        )

    def reset(self):
        self.noise.reset()