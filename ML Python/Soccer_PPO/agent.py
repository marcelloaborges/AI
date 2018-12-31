import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent:

    def __init__(
        self, 
        device,
        key,
        actor_model,
        memory
        ):

        self.DEVICE = device
        self.KEY = key

        self.actor_model = actor_model

        self.memory = memory        

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():                
            action, log_prob, _ = self.actor_model(state)                    
        self.actor_model.train()

        action = action.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach().numpy()

        return action, log_prob

    def step(self, state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward):                
        self.memory.add( state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward )