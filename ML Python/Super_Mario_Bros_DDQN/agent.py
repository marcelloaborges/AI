import torch
import numpy as np
import random

class Agent:

    def __init__(
        self, 
        device,
        cnn,
        model,
        actions_size
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.cnn = cnn
        self.model = model

        self.actions_size = actions_size

    def act(self, state, eps=0.):
        state = torch.from_numpy(state.T.copy()).float().unsqueeze(0).to(self.DEVICE)        

        self.cnn.eval()
        self.model.eval()

        with torch.no_grad():
            state_flatten = self.cnn(state)
            action_values = self.model(state_flatten)

        self.cnn.train()
        self.model.train()
        
        if np.random.uniform() < eps:
            return random.choice(np.arange(self.actions_size))            
        else:
            action = np.argmax(action_values.cpu().data.numpy())
            return action