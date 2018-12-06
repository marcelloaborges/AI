import numpy as np
import random

from model import Model

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Agent:

    def __init__(self, 
        device,
        key,
        model,
        optimizer,                
        n_steps,    
        gamma,            
        beta,
        random_seed):   

        self.DEVICE = device

        self.KEY = key

        # Neural model
        self.model = model
        self.optimizer = optimizer

        # Replay memory        
        self.STATE = 0        
        self.PROB = 1
        self.REWARD = 2
        self.experience = [ [], [], [] ]

        # Hyperparameters        
        self.N_STEPS = n_steps
        self.GAMMA = gamma
        self.BETA = beta

        self.seed = random.seed(random_seed)

        self.t_step = 0

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
                
        self.model.eval()
        with torch.no_grad():
            action, prob, _ = self.model(state)
        self.model.train()  
        
        action = action.cpu().detach().numpy().item()
        prob = prob.cpu().detach().numpy()

        return action, prob

    def step(self, state, prob, reward):        
        # Save experience / reward
        self.experience[self.STATE].append(state)
        self.experience[self.PROB].append(prob)
        self.experience[self.REWARD].append(reward)

        # self.learn()
        self.t_step = (self.t_step + 1) % self.N_STEPS
        if self.t_step == 0:
            self.learn()

    def learn(self):                              
        discounts = self.GAMMA ** np.arange( len( self.experience[self.REWARD] ) )
        rewards = np.asarray( self.experience[self.REWARD] ) * discounts[:,np.newaxis]
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = ( rewards_future - mean[ :, np.newaxis ] ) / std[ :, np.newaxis ]

        old_probs = torch.tensor( self.experience[self.PROB], device=self.DEVICE )
        rewards = torch.tensor( rewards_normalized, dtype=torch.float, device=self.DEVICE )

        states = torch.from_numpy(np.stack(self.experience[self.STATE])).float().to(self.DEVICE)   
        _, new_probs, entropy = self.model(states)
        ratio = new_probs / old_probs

        # entropy = - ( new_probs * torch.log( old_probs + 1.e-10 ) + ( 1.0 - new_probs ) * torch.log( 1.0 - old_probs + 1.e-10 ) )

        loss = - torch.mean( ratio * rewards + self.BETA * entropy )
        # loss = - torch.mean( ratio * rewards )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del loss

        self.BETA *= .995

        self.experience = [ [], [], [] ]      

        # _, log_probs, _ = self.model(states)  

        # discounts = [self.GAMMA**i for i in range(len(self.experience[self.REWARD])+1)]
        # R = sum( [ discount * reward for discount, reward in zip( discounts, self.experience[self.REWARD] ) ] )
        
        # policy_loss = []        
        # for log_prob in log_probs:
        #     policy_loss.append( - log_prob.unsqueeze(0) * R )
        # policy_loss = torch.cat( policy_loss ).sum()        
        
        # self.optimizer.zero_grad()
        # policy_loss.backward()
        # self.optimizer.step()

        # self.experience = [ [], [] ]        
