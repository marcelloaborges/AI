import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:

    def __init__(
        self, 
        device,
        memory,        
        model,
        optimizer,
        n_step,
        batch_size,
        gamma,
        epsilon,
        entropy_weight,
        gradient_clip
        ):

        self.DEVICE = device
        
        # MEMORY
        self.memory = memory

        # NEURAL MODEL
        self.model = model
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

    def step(self, state, hx, cx, action, log_prob, reward):
        self.memory.add( state.T, hx.squeeze(0), cx.squeeze(0), action, log_prob, reward )

        self.t_step = (self.t_step + 1) % self.N_STEP
        if self.t_step == 0:
            self.loss = self._learn()            

        return self.loss

        
    def _learn(self):                    
        states, hxs, cxs, actions, log_probs, rewards, n_exp = self.memory.experiences()


        discount = self.GAMMA**np.arange(n_exp)
        rewards = rewards.squeeze(1) * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)

        hxs = torch.from_numpy(hxs).float().to(self.DEVICE)
        cxs = torch.from_numpy(cxs).float().to(self.DEVICE)

        actions = torch.from_numpy(actions).long().to(self.DEVICE).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE).squeeze(1)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)


        self.model.eval()
        with torch.no_grad():
            _, _, _, values, _, _ = self.model( states, hxs, cxs )
            values = values.detach()
        self.model.train()
                        
        advantages = (rewards - values.squeeze())
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)


        batches = BatchSampler( SubsetRandomSampler( range(0, n_exp) ), self.BATCH_SIZE, drop_last=False)
        
        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]
            sampled_hxs = hxs[batch_indices]
            sampled_cxs = cxs[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages_normalized[batch_indices]            


            _, new_log_probs, entropies, values, _, _ = self.model(sampled_states, sampled_hxs, sampled_cxs, sampled_actions.unsqueeze(1))


            ratio = ( new_log_probs.squeeze() - sampled_log_probs ).exp()

            clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

            policy_loss = torch.min( ratio * sampled_advantages, clip * sampled_advantages )
            policy_loss = - torch.mean( policy_loss )

            entropy = torch.mean(entropies)


            value_loss = F.mse_loss( sampled_rewards, values.squeeze() )


            loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)  


            self.optimizer.zero_grad()                  
            loss.backward()
            # nn.utils.clip_grad_norm_( self.actor_model.parameters(), self.GRADIENT_CLIP )
            # nn.utils.clip_grad_norm_( self.critic_model.parameters(), self.GRADIENT_CLIP )
            self.optimizer.step()


        # self.EPSILON *= 1
        # self.ENTROPY_WEIGHT *= 0.995

        return loss.data
