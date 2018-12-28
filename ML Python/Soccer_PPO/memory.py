import numpy as np
from collections import namedtuple
import random

class Memory:        
    def __init__(self):                        

        self.memory = []
        self.experience = namedtuple('Experience', 
            field_names=['state', 'teammate_state', 'adversary_state', 'adversary_teammate_state', 'action', 'log_prob', 'reward', 'done'])

    def add(self, state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward, done):
        """Add a new experience to memory."""        
        e = self.experience( state, teammate_state, adversary_state, adversary_teammate_state, action, log_prob, reward, done )
        self.memory.append(e)

    def experiences(self):
        states = np.vstack([e.state for e in self.memory if e is not None])
        teammate_states = np.vstack([e.teammate_state for e in self.memory if e is not None])
        adversary_states = np.vstack([e.adversary_state for e in self.memory if e is not None])
        adversary_teammate_states = np.vstack([e.adversary_teammate_state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        log_probs = np.vstack([e.log_prob for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        dones = np.vstack([e.done for e in self.memory if e is not None])
        
        self.clear()

        return states, teammate_states, adversary_states, adversary_teammate_states, actions, log_probs, rewards, dones
    
    def clear(self):
        self.memory.clear()
    
    def __len__(self):    
        return len(self.memory)
