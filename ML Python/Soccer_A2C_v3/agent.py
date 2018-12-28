import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import Memory

class A2CAgent:

    def __init__(
        self, 
        device,
        key,
        model,
        optimizer,
        n_step,
        gamma,
        gae_tau,
        entropy_weight,
        gradient_clip):

        self.DEVICE = device
        self.KEY = key

        self.model = model
        self.optimizer = optimizer

        self.memory = Memory()

        # HYPERPARAMETERS
        self.N_STEP = n_step
        self.GAMMA = gamma
        self.GAE_TAU = gae_tau
        self.ENTROPY_WEIGHT = entropy_weight
        self.GRADIENT_CLIP = gradient_clip

        self.t_step = 0
        self.loss = 0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():                
            action, _, _, _ = self.model(state)                    
        self.model.train()

        action = action.cpu().detach().numpy()[0]
        # log_prob = log_prob.cpu().detach().numpy()

        return action

    def step(self, state, reward, done):                
        self.memory.add( state, reward, 1 - done )

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step != 0:
            return self.loss

        # LEARN
        states, rewards, dones = self.memory.experiences()

        states = torch.from_numpy(states).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        dones = torch.from_numpy(dones).float().to(self.DEVICE)

        _, log_probs, entropies, values = self.model(states)
                        
        # returns = rewards + self.GAMMA * values * dones
        returns = []
        advantage = 0
        advantages = []
        for i in reversed( range(self.N_STEP) ):
            ret = rewards[i] + self.GAMMA * values[i] * dones[i]
            returns.append(ret)

            advantage = ret - values[i]
            
            # td_error = rewards[i] - values[i]
            # advantage = advantage * self.GAE_TAU * self.GAMMA * dones[i] + td_error

            advantages.append( advantage )

        returns = torch.cat(returns)
        advantages = torch.cat( advantages )


        policy_loss = - ( log_probs * advantages ).mean()
        # value_loss = 0.5 * ( returns - values ).pow(2).mean()
        value_loss = 0.5 * F.mse_loss(returns, values)
        entropy_loss = entropies.mean()

        self.optimizer.zero_grad()
        
        loss = policy_loss - self.ENTROPY_WEIGHT * entropy_loss + value_loss
        loss.backward()
        nn.utils.clip_grad_norm_( self.model.parameters(), self.GRADIENT_CLIP )

        self.optimizer.step()        

        self.loss = loss.cpu().detach().numpy()
        return self.loss
