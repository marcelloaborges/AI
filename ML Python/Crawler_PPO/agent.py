import numpy as np
import random
import os

import torch
import torch.optim as optim

class Agent:

    def __init__(self, 
        device,        
        actor_critic_model,        
        shared_memory):

        self.DEVICE = device

        # Actor Network
        self.actor_critic_model = actor_critic_model

        # Replay memory        
        self.shared_memory = shared_memory
    
    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.actor_critic_model.eval()
        with torch.no_grad():
            action, log_prob, _, _ = self.actor_critic_model(state)            
        self.actor_critic_model.train()

        action = action.cpu().data.numpy()
        log_prob = log_prob.cpu().data.numpy()
    
        return np.clip(action, -1, 1), log_prob

    def step(self, keys, states, actions, log_probs, rewards):
        # Save experience / reward        
        for i, key in enumerate(keys):
            self.shared_memory.add( 
                key,
                states[i],
                actions[i],
                log_probs[i],
                rewards[i]
            )