import torch
import numpy as np
import random

class Agent:

    def __init__(
        self, 
        device,
        cnn,
        actor,
        noise
        ):

        self.DEVICE = device

        # NEURAL MODEL
        self.cnn = cnn
        self.actor = actor

        # NOISE
        self.noise = noise

    def act(self, state, add_noise=False):
        state = torch.from_numpy(state.T.copy()).float().unsqueeze(0).to(self.DEVICE)        

        self.cnn.eval()
        self.actor.eval()

        with torch.no_grad():
            state_flatten = self.cnn(state)
            action_values = self.actor(state_flatten).cpu().data.numpy()

        self.cnn.train()
        self.actor.train()
        
        if add_noise:
            action_values += self.noise.sample()

        action = np.argmax(action_values)

        return action

    def reset(self):
        self.noise.reset()