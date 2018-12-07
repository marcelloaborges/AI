import numpy as np

import torch
import torch.nn.functional as F

class Critic:

    def __init__(self, 
        device,        
        model,    
        optimizer,    
        gamma,
        n_step,
        epsilon,
        entropy_weight):   

        self.DEVICE = device        

        # Neural model
        self.model = model
        self.optimizer = optimizer

        # Hyperparameters
        self.GAMMA = gamma
        self.N_STEP = n_step
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight

        self.t_step = 0

    def step(self, actors):
        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:
            for actor in actors:
                self._learn(actor)

    def _learn(self, actor):
        # Get the experiences from the actor and clear the memory
        if not actor.enough_experiences():
            return

        states, actions, actions_probs, rewards, next_states = actor.experiences()

        # Calc the discounted rewards
        discounts = self.GAMMA ** np.arange( len( rewards ) )
        rewards = np.asarray( rewards ) * discounts[:,np.newaxis]
        future_rewards = rewards[::-1].cumsum(axis=0)[::-1]

        # To tensor
        states = torch.from_numpy(states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        actions_probs = torch.from_numpy(actions_probs).float().to(self.DEVICE)
        future_rewards = torch.from_numpy(future_rewards.copy()).float().to(self.DEVICE) ## no copy
        next_states = torch.from_numpy(next_states).float().to(self.DEVICE)

        
        # loss        
        _, new_probs, entropy, values = self.model(states)
        values = values.cpu().detach().numpy()

        # actor       
        advantages = rewards - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.from_numpy(advantages_normalized).float().to(self.DEVICE)
         
        ratio = new_probs / actions_probs

        ratio_clipped = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        objective = torch.min( ratio * advantages_normalized, ratio_clipped * advantages_normalized)
        
        policy_loss = - torch.mean(objective + self.entropy_weight * entropy)

        # critic
        _, _, _, values = self.model(states)
        value_loss = F.mse_loss(future_rewards, values)

        # optimize
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        policy_loss_value = policy_loss.cpu().detach().numpy().squeeze().item()
        value_loss_value = value_loss.cpu().detach().numpy().squeeze().item()

        # print('\rPolicy loss: \t{:.8f} \tValue loss: \t{:.8f}'.format(policy_loss_value, value_loss_value), end="")  
        # if np.isnan(policy_loss_value) or np.isnan(value_loss_value):
            # print("policy_loss_value: {:.6f}, value_loss_value: {:.6f}".format(policy_loss_value, value_loss_value))        

        # the clipping parameter reduces as time goes on
        self.epsilon *= 0.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        self.entropy_weight *= 0.995