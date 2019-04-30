import numpy as np
import random
import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import CriticModel

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:        

    def __init__(self, 
        device,
        actor_model, actor_target, actor_optimizer, 
        critic_model, critic_target, critic_optimizer, 
        shared_memory,
        n_step, batch_size, gamma, TAU):

        self.DEVICE = device

        # Actor Network (w/ Target Network)
        self.actor_model = actor_model
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer

        # Critic Network (w/ Target Network)
        self.critic_model = critic_model
        self.critic_target = critic_target
        self.critic_optimizer = critic_optimizer

        # Shared memory
        self.memory = shared_memory

        # Hyperparameters
        self.N_STEP = n_step
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU        
        
        self.t_step = 0
        self.actor_loss = 0
        self.critic_loss = 0

    def step(self):
        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:            
            self._learn()
        
        return self.actor_loss, self.critic_loss        

    def _learn(self):
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
        
        # keys, states, actions, rewards, next_states, dones = self.memory.sample()
        experiences = self.memory.sample()
        
        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        for key, experience in experiences.items():
            temp_reward = []
            for exp in experience:
                states.append(      exp['state']      )
                actions.append(     exp['action']     )
                temp_reward.append( exp['reward']     )
                next_states.append( exp['next_state'] )
                dones.append(       exp['done']       )

            discount = self.GAMMA**np.arange(len(temp_reward))
            temp_reward = temp_reward * discount
            temp_reward = temp_reward[::-1].cumsum(axis=0)[::-1]

            rewards.extend( temp_reward )


        states      = torch.from_numpy( np.array(states)                                ).float().to(self.DEVICE)
        actions     = torch.from_numpy( np.array(actions)                               ).float().to(self.DEVICE)
        rewards     = torch.from_numpy( np.array(rewards).reshape(-1, 1)                ).float().to(self.DEVICE)
        next_states = torch.from_numpy( np.array(next_states)                           ).float().to(self.DEVICE)
        dones       = torch.from_numpy( np.array(dones).astype(np.uint8).reshape(-1, 1) ).float().to(self.DEVICE)
    

        batches = BatchSampler( SubsetRandomSampler( range(0, len(states)) ), self.BATCH_SIZE, drop_last=False)                        

        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]                        
            sampled_rewards = rewards[batch_indices]
            sampled_next_states = next_states[batch_indices]
            sampled_dones = dones[batch_indices]        

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(sampled_next_states)
            # Q_targets = self.critic_target_model(states, actions)
            Q_targets_next = self.critic_target(sampled_next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = sampled_rewards + (self.GAMMA * Q_targets_next * (1 - sampled_dones))
            # Compute critic loss
            Q_expected = self.critic_model(sampled_states, sampled_actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic_model.parameters(), 0.5)
            self.critic_optimizer.step()

            self.critic_loss = critic_loss.data

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_model(sampled_states)
            actor_loss = - self.critic_model(sampled_states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()    

            self.actor_loss = actor_loss.data

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_model, self.critic_target)
            self.soft_update(self.actor_model, self.actor_target)    

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