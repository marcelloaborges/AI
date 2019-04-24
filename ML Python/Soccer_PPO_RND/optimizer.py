import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:

    def __init__(
        self, 
        device,
        actor_critic_model,
        rnd_target_model,
        rnd_predictor_model,
        optimizer,
        n_step,        
        batch_size,        
        ext_gamma,
        ext_coef,
        int_gamma,
        int_coef,
        epsilon,
        entropy_weight,
        gradient_clip
        ):

        self.DEVICE = device
        

        # NEURAL MODEL
        self.actor_critic_model = actor_critic_model
        self.rnd_target_model = rnd_target_model
        self.rnd_predictor_model = rnd_predictor_model
        self.optimizer = optimizer

        # HYPERPARAMETERS
        self.N_STEP = n_step        
        self.BATCH_SIZE = batch_size                
        self.EXT_GAMMA = ext_gamma
        self.EXT_COEF = ext_coef
        self.INT_GAMMA = int_gamma
        self.INT_COEF = int_coef
        self.GAMMA_N = ext_gamma ** n_step
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip        

        
    def learn(self, memory):                    
        states, actions, log_probs, rewards, n_exp = memory.experiences()


        discount = self.EXT_GAMMA ** np.arange(n_exp)
        rewards = rewards.squeeze(1) * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)        
        actions = torch.from_numpy(actions).long().to(self.DEVICE).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE).squeeze(1)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)


        self.actor_critic_model.eval()
        with torch.no_grad():
            _, _, _, values = self.actor_critic_model( states )
            values = values.detach()
        self.actor_critic_model.train()
                        
        advantages = (rewards - values.squeeze()).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)        


        batches = BatchSampler( SubsetRandomSampler( range(0, n_exp) ), self.BATCH_SIZE, drop_last=False)

        losses = []
        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)

            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages_normalized[batch_indices]            


            _, new_log_probs, entropies, values = self.actor_critic_model(sampled_states, sampled_actions)


            ratio = ( new_log_probs - sampled_log_probs ).exp()

            clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

            policy_loss = torch.min( ratio * sampled_advantages, clip * sampled_advantages )
            policy_loss = - torch.mean( policy_loss )

            entropy = torch.mean(entropies)

            
            value_loss = F.mse_loss( sampled_rewards, values.squeeze() )


            # RND Reward
            rnd_target = self.rnd_target_model( sampled_states )
            rnd_target.detach()
            rnd_predict = self.rnd_predictor_model( sampled_states )
            

            advantages_rnd = ( torch.pow( rnd_predict - rnd_target, 2 ).sum(1) / 2 ).clip(-5, 5)
            # advantages_rnd_normalized = (advantages_rnd - advantages_rnd.mean()) / (advantages_rnd.std() + 1.0e-10)
            # advantages_rnd_normalized = torch.tensor(advantages_rnd_normalized).float().to(self.DEVICE)

            
            # combined_advantages = self.EXT_COEF * sampled_advantages + self.INT_COEF * advantages_rnd


            loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT) + self.INT_COEF * advantages_rnd


            self.optimizer.zero_grad()                  
            loss.backward()
            # nn.utils.clip_grad_norm_( self.actor_model.parameters(), self.GRADIENT_CLIP )
            # nn.utils.clip_grad_norm_( self.critic_model.parameters(), self.GRADIENT_CLIP )
            self.optimizer.step()


            losses.append( loss.data )

        # self.EPSILON *= 1
        # self.ENTROPY_WEIGHT *= 0.995

        return np.average( losses )
