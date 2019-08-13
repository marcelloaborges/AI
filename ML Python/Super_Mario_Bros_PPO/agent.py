import torch
import numpy as np
import random

class Agent:

    def __init__(
        self, 
        device,
        cnn,
        actor        
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.cnn = cnn
        self.actor = actor        

    def act(self, state):
        state = torch.from_numpy(state.T.copy()).float().unsqueeze(0).to(self.DEVICE)        

        self.cnn.eval()
        self.actor.eval()

        with torch.no_grad():
            state_flatten = self.cnn(state)
            action, log_prob, _ = self.actor(state_flatten)

        self.cnn.train()
        self.actor.train()
        
        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()

        return action, log_prob