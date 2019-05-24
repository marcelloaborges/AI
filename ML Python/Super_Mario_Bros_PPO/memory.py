import numpy as np
from collections import namedtuple
import random

class Memory:        
    def __init__(self):                        

        self.memory = []
        self.experience = namedtuple('Experience', 
            field_names=['state', 'hx', 'cx', 'action', 'log_prob', 'reward'])

    def add(self, state, hx, cx, action, log_prob, reward):
        """Add a new experience to memory."""        
        e = self.experience( state, hx, cx, action, log_prob, reward )
        self.memory.append(e)

    def experiences(self, clear=True):
        states = np.stack([e.state for e in self.memory if e is not None])
        hxs = np.stack([e.hx for e in self.memory if e is not None])
        cxs = np.stack([e.cx for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        log_probs = np.vstack([e.log_prob for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
                
        n_exp = len(self)

        if clear:
            self.clear()

        return states, hxs, cxs, actions, log_probs, rewards, n_exp
    
    def delete(self, i):
        del self.memory[i]

    def clear(self):
        self.memory.clear()
    
    def __len__(self):    
        return len(self.memory)
