import numpy as np

import torch
import torch.nn.functional as F

class Critic:

    def __init__(self, 
        device,        
        actor_model,    
        critic_model,
        optimizer,           
        batch_size,                    
        epsilon,
        beta):

        self.DEVICE = device        

        # Neural model
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer        

        # Hyperparameters
        self.BATCH_SIZE = batch_size        
        self.epsilon = epsilon
        self.beta = beta
        
    def step(self, memory):        
        if memory.enough_experiences(self.BATCH_SIZE):            
            experiences = memory.experiences()
            self._learn(experiences)
            
    def _learn(self, experiences):
        # Get the experiences from the actor and clear the memory        
        # The rewards have the discount already applied
        states, teammate_states, actions, actions_probs, rewards = experiences

        # To tensor
        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        actions_probs = torch.from_numpy(actions_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)        

        
        # loss        
        _, new_probs, entropy = self.actor_model(states, actions)        
        values = self.critic_model(states, teammate_states)

        # actor       
        advantages = rewards - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)                
         
        ratio = (new_probs - actions_probs).exp()

        ratio_clipped = torch.clamp( ratio, 1 - self.epsilon, 1 + self.epsilon )
        objective = torch.min( ratio * advantages_normalized, ratio_clipped * advantages_normalized )        

        policy_loss = - torch.mean( objective )
        entropy = torch.mean( entropy )

        # critic        
        value_loss = F.mse_loss(rewards, values)

        # optimize
        loss = policy_loss + 0.5 * value_loss + entropy * self.beta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # policy_loss_value = policy_loss.cpu().detach().numpy().squeeze().item()
        # value_loss_value = value_loss.cpu().detach().numpy().squeeze().item()

        # print('\rPolicy loss: \t{:.8f} \tValue loss: \t{:.8f}'.format(policy_loss_value, value_loss_value), end="")  
        # if np.isnan(policy_loss_value) or np.isnan(value_loss_value):
            # print("policy_loss_value: {:.6f}, value_loss_value: {:.6f}".format(policy_loss_value, value_loss_value))        

        
        self.epsilon *= 0.999
        self.beta *= 0.995