import random
import numpy as np

import torch

class Agent:

    def __init__(self, 
            device,
            key,
            model,
            shared_memory, 
            action_size, 
            eps_start, eps_end, eps_steps):
        
        self.DEVICE = device                    
        self.KEY = key

        self.model = model
        
        self.shared_memory = shared_memory

        # HYPERPARAMETERS        
        self.ACTION_SIZE = action_size        
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_STEPS = eps_steps

        self.t_step = 0
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()

        temp = np.random.uniform()
        eps = self._get_epsilon()
        if temp < eps:
            return random.choice(np.arange(self.ACTION_SIZE))
        else:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
            
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.shared_memory.add(state, action, reward, next_state, done)        

    def _get_epsilon(self):
        self.t_step += 1
        if(self.t_step >= self.EPS_STEPS):
            return self.EPS_END
        else:
            return self.EPS_START + self.t_step * (self.EPS_END - self.EPS_START) / self.EPS_STEPS

