import threading

import torch
import random
import numpy as np
import copy

from v1_DDPG.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():    

    def __init__(self,
        id,
        brain,
        gamma,
        gamma_n, 
        n_step_return):                
        
        self.id = id
        self.brain = brain

        self.GAMMA = gamma
        self.GAMMA_N = gamma_n
        self.N_STEP_RETURN = n_step_return        

        self.R = 0
        self.memory = []

    def compute_epsilon(self):
        return 0.005

    def act(self, state):        
        action, critic = self.brain.forward(state)                        

        return action, critic

    def step(self, state, action, reward, next_state, done):        
        def get_sample(memory, n):
            state, action, _, _, _  = memory[0]
            _, _, _, next_state, done = memory[n-1]
            
            return state, action, self.R, next_state, done             
        
        self.memory.append( (state, action, reward, next_state, done) )
        
        self.R = ( self.R + reward * self.GAMMA_N ) / self.GAMMA
        
        if done:
            while len(self.memory) > 0:
                n = len(self.memory)
                state, action, reward, next_state, done = get_sample(self.memory, n)
                self.brain.experience_push(state, action, reward, next_state, done)
                
                self.R = ( self.R - self.memory[0][2] ) / self.GAMMA
                self.memory.pop(0)
            
            self.R = 0
            
        if len(self.memory) >= self.N_STEP_RETURN:
            state, action, reward, next_state, done = get_sample(self.memory, self.N_STEP_RETURN)
            self.brain.experience_push(state, action, reward, next_state, done)
            
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)	

