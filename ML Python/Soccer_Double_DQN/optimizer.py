import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

class Optimizer:

    def __init__(self, 
        device,
        memory,
        model,
        target_model,
        optimizer,
        gamma, tau):        

        self.DEVICE = device

        self.memory = memory

        self.model = model
        self.target_model = target_model

        self.optimizer = optimizer

        # Hyperparameters
        self.GAMMA = gamma
        self.TAU = tau
                

    def step(self):
        # If enough samples are available in memory, get random subset and learn                
        experiences = self.memory.sample()
        if not experiences:
            return

        states, actions, rewards, next_states, dones = experiences  

        # To tensor
        states = torch.from_numpy(states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        next_states = torch.from_numpy(next_states).float().to(self.DEVICE)
        dones = torch.from_numpy(dones).float().to(self.DEVICE)        

        # Calculate the loss
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + self.GAMMA * Q_targets_next * (1 - dones)

        Q_value = self.model(states).gather(1, actions)
                
        loss = F.smooth_l1_loss(Q_value, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target model
        self._soft_update_target_model()
    
    def _soft_update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)