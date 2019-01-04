import numpy as np
import random
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import CriticModel

class Optimizer:        

    def __init__(self, 
        device,
        state_size,
        action_size,        
        lr,
        weight_decay,
        memory,
        n_step,
        gamma, TAU,
        checkpoint_folder):

        self.DEVICE = device

        self.checkpoint_file = checkpoint_folder + 'checkpoint_critic.pth'

        # Critic Network (w/ Target Network)
        self.critic_model = CriticModel(state_size, action_size).to(self.DEVICE)
        self.critic_target_model = CriticModel(state_size, action_size).to(self.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=lr, weight_decay=weight_decay)

        self.critic_model.load(self.checkpoint_file)
        self.critic_target_model.load(self.checkpoint_file)

        # Shared memory
        self.memory = memory

        # Hyperparameters
        self.N_STEP = n_step
        self.GAMMA = gamma
        self.TAU = TAU        
        
        self.t_step = 0
        self.critic_loss = 0

    def step(self, agent):
        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step != 0:
            return self.critic_loss

        # Learn, if enough samples are available in memory  
        if not self.memory.enough_experiences():
            return self.critic_loss

        self.learn(agent)

        return self.critic_loss

    def learn(self, agent):
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
          
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.from_numpy(states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        next_states = torch.from_numpy(next_states).float().to(self.DEVICE)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.DEVICE)
        
        # weights = torch.tensor(weights, device=self.DEVICE, dtype=torch.float)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = agent.actor_target_model(next_states)
        # Q_targets = self.critic_target_model(states, actions)
        Q_targets_next = self.critic_target_model(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_model(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_model.parameters(), 0.5)
        self.critic_optimizer.step()

        self.critic_loss = critic_loss        

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = agent.actor_model(states)
        actor_loss = - self.critic_model(states, actions_pred).mean()
        # Minimize the loss
        agent.actor_optim.zero_grad()
        actor_loss.backward()
        agent.actor_optim.step()        

        # ------------------------- update experiences ------------------------- #
        # exp_loss = (Q_expected - Q_targets).detach().squeeze().abs().cpu().numpy().tolist()
        # self.memory.update_priorities(indices, exp_loss)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_model, self.critic_target_model)
        self.soft_update(agent.actor_model, agent.actor_target_model)        

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
        self.critic_model.checkpoint( self.checkpoint_file )