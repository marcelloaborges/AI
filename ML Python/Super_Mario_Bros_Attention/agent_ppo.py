import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_buffer import MemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent:

    def __init__(
        self, 
        device,
        seq_len,
        batch_size,
        gamma, epsilon, entropy_weight, gradient_clip,
        attention_model, actor_model, critic_model,
        optimizer,        
        checkpoint_actor,
        checkpoint_critic
        ):

        self.DEVICE = device

        # HYPERPARAMETERS                                   
        self.SEQ_LEN = seq_len        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.GAMMA_N = gamma ** seq_len
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip 

        # NEURAL MODEL
        self.attention_model = attention_model
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.attention_model.eval()

        self.optimizer = optimizer

        self.CHECKPOINT_ACTOR = checkpoint_actor
        self.CHECKPOINT_CRITIC = checkpoint_critic

        # MEMORY
        self.memory = MemoryBuffer()

        # AUX        
        self.t_step = 0
        self.loss = 0
    
    def act(self, state):        
        state = torch.tensor(state).unsqueeze(0).float().to(self.DEVICE)
        
        self.actor_model.eval()

        with torch.no_grad():            
            encoded = self.attention_model(state)
            action, log_prob, _ = self.actor_model(encoded[:,-1:].squeeze(1))
                    
        self.actor_model.train()        
                
        action = action.cpu().data.numpy().item()
        log_prob = log_prob.cpu().data.numpy().item()
            
        return action, log_prob

    def step(self, state, action, log_prob, reward):
        self.memory.add( state, action, log_prob, reward )
            
        if len(self.memory) == self.BATCH_SIZE:
            self.loss = self._learn()

        return self.loss

    def _learn(self):        
        states, actions, log_probs, rewards = self.memory.experiences()

        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1.0e-10)

        discount = self.GAMMA**np.arange(self.BATCH_SIZE)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        states    = torch.from_numpy( states                ).float().to(self.DEVICE)
        actions   = torch.from_numpy( actions               ).float().to(self.DEVICE)
        log_probs = torch.from_numpy( log_probs             ).float().to(self.DEVICE)
        rewards   = torch.from_numpy( rewards_future.copy() ).float().to(self.DEVICE)                

        # PPO

        self.critic_model.eval()
        with torch.no_grad():
            encoded = self.attention_model(states)[:,-1:].squeeze(1)
            values = self.critic_model( encoded ).detach()
        self.critic_model.train()

        advantages = (rewards - values.squeeze()).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)

        batches = BatchSampler( SubsetRandomSampler( range(0, self.BATCH_SIZE) ), int(self.BATCH_SIZE / 4), drop_last=False)

        avg_loss = (0.0, 0.0)
        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]            
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages_normalized[batch_indices]            


            encoded = self.attention_model(sampled_states)[:,-1:].squeeze(1)


            _, new_log_probs, entropies = self.actor_model(encoded, sampled_actions)

            ratio = ( new_log_probs - sampled_log_probs ).exp()

            clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

            policy_loss = torch.min( ratio * sampled_advantages, clip * sampled_advantages )
            policy_loss = - torch.mean( policy_loss )

            entropy = torch.mean(entropies)

            
            values = self.critic_model( encoded ).squeeze()
            value_loss = F.mse_loss( sampled_rewards, values )


            loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)  


            l2_factor = 1e-8

            l2_reg_actor = None
            for W in self.actor_model.parameters():
                if l2_reg_actor is None:
                    l2_reg_actor = W.norm(2)
                else:
                    l2_reg_actor = l2_reg_actor + W.norm(2)

            l2_reg_actor = l2_reg_actor * l2_factor

            loss += l2_reg_actor

            l2_reg_critic = None
            for W in self.critic_model.parameters():
                if l2_reg_critic is None:
                    l2_reg_critic = W.norm(2)
                else:
                    l2_reg_critic = l2_reg_critic + W.norm(2)

            l2_reg_critic = l2_reg_critic * l2_factor

            loss += l2_reg_critic


            self.optimizer.zero_grad()                  
            loss.backward()
            # nn.utils.clip_grad_norm_( self.actor_model.parameters(), self.GRADIENT_CLIP )
            # nn.utils.clip_grad_norm_( self.critic_model.parameters(), self.GRADIENT_CLIP )
            self.optimizer.step()

            avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)        

        return (avg_loss[0]/avg_loss[1]).cpu().data.numpy().item()
   
    def checkpoint(self):                
        self.actor_model.checkpoint(self.CHECKPOINT_ACTOR)
        self.critic_model.checkpoint(self.CHECKPOINT_CRITIC)
