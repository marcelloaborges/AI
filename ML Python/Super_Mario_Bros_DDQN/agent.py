import numpy as np
import random

import torch

class Agent:

    def __init__(
        self, 
        device,
        encoder,
        ddqn_model,
        actions_size
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.encoder = encoder
        self.ddqn_model = ddqn_model

        self.actions_size = actions_size

    def act(self, state, hx, cx, eps=0.):                
        state = torch.tensor(state).float().unsqueeze(0).to(self.DEVICE)
        hx    = torch.from_numpy( hx ).float().to(self.DEVICE)
        cx    = torch.from_numpy( cx ).float().to(self.DEVICE)

        self.encoder.eval()
        self.ddqn_model.eval()

        with torch.no_grad():
            encoded_state, _ = self.encoder(state)

            action_values, nhx, ncx = self.ddqn_model(encoded_state, hx, cx)

        self.encoder.train()
        self.ddqn_model.train()

        action_values = action_values.cpu().data.numpy()
        nhx = nhx.cpu().data.numpy()
        ncx = ncx.cpu().data.numpy()
        
        action = None
        if np.random.uniform() < eps:
            action = random.choice(np.arange(self.actions_size))            
        else:
            action = np.argmax(action_values)
            
        return action, nhx, ncx