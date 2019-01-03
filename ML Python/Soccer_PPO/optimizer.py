import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from memory import Memory

class Optimizer:

    def __init__(
        self, 
        device,        
        actor_model,
        critic_model,
        optimizer,        
        n_step,
        batch_size,
        gamma,
        epsilon,
        entropy_weight,
        gradient_clip):

        self.DEVICE = device        

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer        

        # HYPERPARAMETERS
        self.N_STEP = n_step
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip
        
        self.t_step = 0
        self.loss = 0

    def step(self, memory ):
        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step != 0:
            return self.loss

        # LEARN
        states, teammate_states, adversary_states, adversary_teammate_states, actions, log_probs, rewards, n_exp = memory.experiences()

        
        discount = self.GAMMA**np.arange(n_exp).reshape(-1, 1)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=1)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        adversary_states = torch.from_numpy(adversary_states).float().to(self.DEVICE)
        adversary_teammate_states = torch.from_numpy(adversary_teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)


        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
                        

        advantages = (rewards - values).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)


        _, new_log_probs, entropies = self.actor_model(states, actions)

        ratio = ( new_log_probs - log_probs ).exp()
        clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )
        
        policy_loss = torch.min( ratio * advantages, clip * advantages )
        policy_loss = - torch.mean( policy_loss )

        entropy = torch.mean(entropies)
        
        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
        value_loss = F.mse_loss( rewards, values )


        self.optimizer.zero_grad()

        loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)        
        loss.backward()
        # nn.utils.clip_grad_norm_( self.model.parameters(), self.GRADIENT_CLIP )

        self.optimizer.step()


        self.loss = loss.cpu().detach().numpy()

        return self.loss

    def step2(self, memory ):
        # LEARN
        states, teammate_states, adversary_states, adversary_teammate_states, actions, log_probs, rewards, n_exp = memory.experiences()

        
        discount = self.GAMMA**np.arange(n_exp).reshape(-1, 1)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=1)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        adversary_states = torch.from_numpy(adversary_states).float().to(self.DEVICE)
        adversary_teammate_states = torch.from_numpy(adversary_teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)


        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
                        

        advantages = (rewards - values).cpu().detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)


        batches = BatchSampler( SubsetRandomSampler( range(0, n_exp) ), self.BATCH_SIZE, drop_last=False)

        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]
            sampled_teammate_states = teammate_states[batch_indices]
            sampled_adversary_states = adversary_states[batch_indices]
            sampled_adversary_teammate_states = adversary_teammate_states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages_normalized[batch_indices]            


            _, new_log_probs, entropies = self.actor_model(sampled_states, sampled_actions)


            ratio = ( new_log_probs - sampled_log_probs ).exp()
            clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )
            
            policy_loss = torch.min( ratio * sampled_advantages, clip * sampled_advantages )
            policy_loss = - torch.mean( policy_loss )

            entropy = torch.mean(entropies)
            
            values = self.critic_model( torch.cat( (sampled_states, sampled_teammate_states, sampled_adversary_states, sampled_adversary_teammate_states), dim=1 ) )
            value_loss = F.mse_loss( sampled_rewards, values )


            self.optimizer.zero_grad()

            loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)        
            loss.backward()
            # nn.utils.clip_grad_norm_( self.model.parameters(), self.GRADIENT_CLIP )

            self.optimizer.step()


            self.loss = loss.cpu().detach().numpy()

        return self.loss
