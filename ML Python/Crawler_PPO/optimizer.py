import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:        

    def __init__(self, 
        device,
        actor_critic_model, actor_critic_optimizer,         
        shared_memory,
        n_step, gamma, batch_size, epsilon, entropy_weight, gradient_clip):

        self.DEVICE = device

        # Actor Critic Network
        self.actor_critic_model = actor_critic_model
        self.actor_critic_optimizer = actor_critic_optimizer

        # Shared memory
        self.memory = shared_memory

        # Hyperparameters
        self.N_STEP = n_step
        self.GAMMA = gamma  
        self.BATCH_SIZE = batch_size      
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip        

        self.t_step = 0         
        self.loss = 0

    def step(self):
        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:          
            self._learn()  

        return self.loss

    def _learn(self):
        experiences = self.memory.sample()

        states     = []
        actions    = []
        log_probs  = []
        rewards    = []
        advantages = []

        for key, experience in experiences.items():
            temp_rewards = []
            values = []

            for exp in experience:                
                states.append       ( exp['state']    )
                actions.append      ( exp['action']   )
                log_probs.append    ( exp['log_prob'] )
                temp_rewards.append ( exp['reward']   )
                values.append       ( exp['value']    )

            discount = self.GAMMA**np.arange( len(temp_rewards) )
            temp_rewards = np.vstack( temp_rewards ) * discount.reshape( -1, 1 )
            temp_rewards = temp_rewards[::-1].cumsum(axis=0)[::-1]

            temp_advantages = temp_rewards - values
            temp_advantages = (temp_advantages - temp_advantages.mean()) / (temp_advantages.std() + 1.0e-10)

            rewards.extend( temp_rewards )
            advantages.extend( temp_advantages )

        states = np.array( states )
        actions = np.array( actions )
        log_probs = np.array( log_probs )
        rewards = np.array( rewards )
        advantages = np.array( advantages )

        states = torch.from_numpy(states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).float().to(self.DEVICE)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        advantages = torch.from_numpy(advantages).float().to(self.DEVICE)
        

        batches = BatchSampler( SubsetRandomSampler( range(0, len(advantages) ) ), self.BATCH_SIZE, drop_last=False)

        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages[batch_indices]     


            _, new_log_probs, entropies, values = self.actor_critic_model(sampled_states, sampled_actions)


            ratio = ( new_log_probs - sampled_log_probs ).exp()

            clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

            policy_loss = torch.min( ratio * sampled_advantages, clip * sampled_advantages )
            policy_loss = - torch.mean( policy_loss )

            entropy = torch.mean(entropies)


            value_loss = F.mse_loss( sampled_rewards, values )


            loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)  


            self.actor_critic_optimizer.zero_grad()                  
            loss.backward()
            nn.utils.clip_grad_norm_( self.actor_critic_model.parameters(), self.GRADIENT_CLIP )
            self.actor_critic_optimizer.step()


            self.loss = loss.data
        
        # self.EPSILON *= 1
        # self.ENTROPY_WEIGHT *= 0.995