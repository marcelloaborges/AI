import numpy as np
from collections import namedtuple
import random

class Memory:        
    def __init__(self):                        

        self.memory = []        
        self.experience = namedtuple("Experience", field_names=["state", "action", "action_prob", "reward", "next_state"])

    def add(self, state, action, action_prob, reward, next_state):
        """Add a new experience to memory."""        
        e = self.experience(state, action, action_prob, reward, next_state)
        self.memory.append(e)

    def enough_experiences(self, batch_size):
        if len(self.memory) < batch_size:
            return False

        return True

    def experiences(self):        
        states = np.vstack([e.state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        actions_probs = np.vstack([e.action_prob for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        next_states = np.vstack([e.next_state for e in self.memory if e is not None])

        self.memory.clear()

        return states, actions, actions_probs, rewards, next_states    
    
    def __len__(self):    
        return len(self.memory)
