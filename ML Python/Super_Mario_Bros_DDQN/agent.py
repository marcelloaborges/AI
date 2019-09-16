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

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)        

        return z

    def act(self, state, eps=0.):                
        state = torch.tensor(state).float().unsqueeze(0).to(self.DEVICE)

        self.encoder.eval()
        self.ddqn_model.eval()

        with torch.no_grad():
            mu, logvar = self.encoder(state)
            encoded_state = self._reparameterize(mu, logvar)

            action_values = self.ddqn_model(encoded_state)

        self.encoder.train()
        self.ddqn_model.train()
        
        if np.random.uniform() < eps:
            return random.choice(np.arange(self.actions_size))            
        else:
            action = np.argmax(action_values.cpu().data.numpy())
            return action