import numpy as np
import random
import os

from model import ActorNN

import torch
import torch.nn.functional as F
import torch.optim as optim

class Actor:

    def __init__(self, 
        device,
        key,
        player_type,
        state_size, action_size, random_seed,
        memory, noise,
        lr, weight_decay,
        checkpoint_folder = './'):   

        self.DEVICE = device

        self.KEY = key

        self.state_size = state_size        
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.LR = lr        
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network (w/ Target Network)
        self.local = ActorNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.target = ActorNN(state_size, action_size, random_seed).to(self.DEVICE)
        self.optimizer = optim.Adam(self.local.parameters(), lr=self.LR)

        self.checkpoint_full_name = self.CHECKPOINT_FOLDER + 'checkpoint_' + player_type + '_' + str(self.KEY) + '.pth'
        if os.path.isfile(self.checkpoint_full_name):
            self.local.load_state_dict(torch.load(self.checkpoint_full_name))
            self.target.load_state_dict(torch.load(self.checkpoint_full_name))

        # Replay memory        
        self.memory = memory

        # Noise process
        self.noise = noise

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.local.eval()
        with torch.no_grad():
            action_values = self.local(state).cpu().data.numpy()
        self.local.train()                

        if add_noise:
            action_values += self.noise.sample()

        action = np.argmax(action_values)                
    
        return action, action_values
        # return np.clip(action, 0, 1), action_values

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
     
    def checkpoint(self):
        torch.save(self.local.state_dict(), self.checkpoint_full_name)