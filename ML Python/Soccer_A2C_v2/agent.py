import numpy as np
import torch

from memory import Memory

class Agent:

    def __init__(self,
        device,
        key,
        model):

        self.DEVICE = device
        self.KEY = key                

        # Neural model
        self.model = model        
        

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
                
        self.model.eval()
        with torch.no_grad():
            action, log_prob, _ = self.model(state)            
        self.model.train()

        action = action.cpu().detach().numpy().squeeze(1)
        log_prob = log_prob.cpu().detach().numpy().squeeze(1)

        return action, log_prob
