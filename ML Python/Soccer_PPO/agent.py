import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import Memory

class PPOAgent:

    def __init__(
        self, 
        device,
        key,
        actor_model,
        critic_model,
        optimizer,
        n_step,
        gamma,
        epsilon,
        entropy_weight,
        gradient_clip):

        self.DEVICE = device
        self.KEY = key

        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer

        self.memory = Memory()

        # HYPERPARAMETERS
        self.N_STEP = n_step
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip

        self.t_step = 0
        self.loss = 0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():                
            action, log_prob, _ = self.actor_model(state)                    
        self.actor_model.train()

        action = action.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach().numpy()

        return action, log_prob

    def step(self, state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward, done):                
        self.memory.add( state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward, 1 - done )

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step != 0:
            return self.loss

        # LEARN
        states, teammate_states, adversary_states, adversary_teammate_states, actions, log_probs, rewards, dones = self.memory.experiences()

        
        discount = self.GAMMA**np.arange(self.N_STEP).reshape(-1, 1)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=1)[::-1]


        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        adversary_states = torch.from_numpy(adversary_states).float().to(self.DEVICE)
        adversary_teammate_states = torch.from_numpy(adversary_teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        log_probs = torch.from_numpy(log_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.DEVICE)
        dones = torch.from_numpy(dones).float().to(self.DEVICE)


        _, new_log_probs, entropies = self.actor_model(states, actions)
        values = self.critic_model( torch.cat( (states, teammate_states, adversary_states, adversary_teammate_states), dim=1 ) )
                        

        advantages = (rewards - values).cpu().detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)


        ratio = ( new_log_probs - log_probs ).exp()
        clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )
        
        policy_loss = torch.min( ratio * advantages_normalized, clip * advantages_normalized )
        policy_loss = - torch.mean( policy_loss )

        entropy = torch.mean(entropies)
        
        value_loss = F.mse_loss(rewards, values)


        self.optimizer.zero_grad()

        loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)        
        loss.backward()
        # nn.utils.clip_grad_norm_( self.model.parameters(), self.GRADIENT_CLIP )

        self.optimizer.step()


        self.loss = loss.cpu().detach().numpy()
        return self.loss
