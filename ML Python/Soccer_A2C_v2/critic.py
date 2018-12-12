import numpy as np

import torch
import torch.nn.functional as F

class Critic:

    def __init__(self, 
        device,        
        model,    
        optimizer,   
        shared_memory, 
        batch_size,                    
        epsilon,
        beta):

        self.DEVICE = device        

        # Neural model
        self.model = model
        self.optimizer = optimizer

        # Shared memory
        self.shared_memory = shared_memory

        # Hyperparameters
        self.BATCH_SIZE = batch_size        
        self.epsilon = epsilon
        self.beta = beta
        
    def step(self):
        if self.shared_memory.enough_experiences(self.BATCH_SIZE):
            self._learn()
            
    def _learn(self):
        # Get the experiences from the actor and clear the memory        
        # The rewards have the discount already applied
        states, actions, actions_probs, rewards, next_states = self.shared_memory.experiences()      

        # To tensor
        states = torch.from_numpy(states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        actions_probs = torch.from_numpy(actions_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        next_states = torch.from_numpy(next_states).float().to(self.DEVICE)

        
        # loss        
        _, new_probs, entropy, values = self.model(states, actions)        

        # actor       
        advantages = rewards - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = advantages
         
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

        
        self.epsilon *= 0.995
        self.beta *= 0.995