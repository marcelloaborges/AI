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

    def act(self, state, hx, cx, eps=0.):
        state = torch.from_numpy( state.T.copy() ).float().unsqueeze(0).to(self.DEVICE)        
        hx = torch.from_numpy( hx ).float().to(self.DEVICE)
        cx = torch.from_numpy( cx ).float().to(self.DEVICE)

        self.model.eval()
        with torch.no_grad():
            action_values, hx, cx = self.model(state, hx, cx)
        self.model.train()
                
        action = None
        if np.random.uniform() < eps:
            action = random.choice(np.arange(self.actions_size))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
          
        hx = hx.cpu().data.numpy()
        cx = cx.cpu().data.numpy()
        
        return action, hx, cx
