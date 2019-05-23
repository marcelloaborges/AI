import torch
import numpy as np
import random

class Agent:

    def __init__(
        self, 
        device,
        model,
        actions_size
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.model = model

        self.actions_size = actions_size

    def act(self, state, hidden, eps=0.):
        state = torch.from_numpy( state.T.copy() ).float().unsqueeze(0).to(self.DEVICE)        
        hidden = torch.from_numpy( hidden ).float().to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():
            action_values, hidden = self.model(state, hidden)
        self.model.train()
                
        action = None
        if np.random.uniform() < eps:
            action = random.choice(np.arange(self.actions_size))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
          
        hidden = hidden.cpu().data.numpy()
        
        return action, hidden
