import numpy as np
import random

from model import ActorCriticNN
from buffer import Buffer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Actor:

    def __init__(self, 
        device,
        key,
        model,
        shared_buffer,         
        action_size,
        gamma,
        gamma_n,
        n_steps,        
        random_seed):   

        self.DEVICE = device

        self.KEY = key
        
        self.action_size = action_size        

        # Neural model
        self.model = model

        # Replay memory        
        self.buffer = Buffer(device, n_steps, random_seed)
        self.shared_buffer = shared_buffer

        # Hyperparameters
        self.action_size = action_size
        self.GAMMA = gamma
        self.GAMMA_N = gamma_n 
        self.N_STEPS = n_steps

        self.seed = random.seed(random_seed)

        self.R = 0

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            action, _, _, _ = self.model(state)
        self.model.train()  

        action = action.cpu().detach().numpy().item()

        return action

    def step(self, state, action, reward, next_state, done):        
        # Save experience / reward
                
        def get_sample(buffer, n):
            exp = buffer.get_experience(0)
            last_exp = buffer.get_experience(n-1)
            
            return exp.state, exp.action, self.R, last_exp.next_state, last_exp.done

        one_hot_action = np.zeros(self.action_size)
        one_hot_action[action] = 1

        self.buffer.add(state, one_hot_action, reward, next_state, done)

        self.R = ( self.R + reward * self.GAMMA_N ) / self.GAMMA        

        # if terminal state, empty the buffer
        if done:
            while len(self.buffer) > 0:
                n = len(self.buffer)
                s, a, r, s_, d = get_sample(self.buffer, n)
                self.shared_buffer.add(s, a, r, s_, d)
                exp = self.buffer.get_experience(0)
                self.R = ( self.R - exp.reward ) / self.GAMMA
                self.buffer.memory.pop(0)
                
            self.R = 0            
     
        # if n_steps calculate reward for the path and store
        if len(self.buffer) >= self.N_STEPS:
            s, a, r, s_, d = get_sample(self.buffer, self.N_STEPS)
            self.shared_buffer.add(s, a, r, s_, d)
            
            exp = self.buffer.get_experience(0)
            self.R = self.R - exp.reward
            self.buffer.memory.pop(0)