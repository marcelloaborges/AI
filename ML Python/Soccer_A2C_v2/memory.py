import numpy as np
from collections import namedtuple
import random

class Memory:        
    def __init__(self):                        

        self.memory = []
        self.experience = namedtuple("Experience", 
            field_names=["state", "teammate_state", "action", "action_prob", "reward"])

    def add(self, state, teammate_state, action, action_prob, reward):
        """Add a new experience to memory."""        
        e = self.experience( state, teammate_state, action, action_prob, reward )
        self.memory.append(e)

    def enough_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        return True

    def experiences(self):
        states = np.vstack([e.state for e in self.memory if e is not None])        
        teammate_states = np.vstack([e.teammate_state for e in self.memory if e is not None])        
        actions = np.vstack([e.action for e in self.memory if e is not None])
        actions_probs = np.vstack([e.action_prob for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])

        self.memory.clear()

        return states, teammate_states, actions, actions_probs, rewards
    
    def __len__(self):    
        return len(self.memory)
