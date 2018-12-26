import numpy as np
import random
import os

from model import CriticModel

import torch
import torch.nn.functional as F
import torch.optim as optim

class Critic:        

    def __init__(self, 
        device,    
        local_model, target_model, optimizer,    
        gamma, TAU,
        checkpoint_folder = './'):        

        self.DEVICE = device

        # Hyperparameters
        self.GAMMA = gamma
        self.TAU = TAU
        
        self.CHECKPOINT_FOLDER = checkpoint_folder
        
        # Critic Network (w/ Target Network)
        self.local = local_model
        self.target = target_model
        self.optimizer = optimizer

        if os.path.isfile(self.CHECKPOINT_FOLDER):
            self.local.load(self.CHECKPOINT_FOLDER)
            self.target.load(self.CHECKPOINT_FOLDER)

        self.t_step = 0

    def step(self, actor_local, actor_target, actor_optimizer, memory):
        self.t_step = (self.t_step + 1) % 4 #self.UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory  
            experiences = memory.sample()
            if not experiences:
                return

            self.learn(actor_local, actor_target, actor_optimizer, experiences)

    def learn(self, actor_local, actor_target, actor_optimizer, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
  
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        _, probs_next = actor_target(next_states)
        Q_targets_next = self.target(next_states, probs_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.local.parameters(), 1)
        self.optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        _, probs_pred = actor_local(states)
        actor_loss = - self.local(states, probs_pred).mean()
        # Minimize the loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local, self.target)
        self.soft_update(actor_local, actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def checkpoint(self):          
        self.local.checkpoint(self.CHECKPOINT_FOLDER)  

