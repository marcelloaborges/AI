import numpy as np
from collections import deque
import random

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size

        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""            
        e = {
            "state" : state,
            "action" : action,
            "reward" : reward,
            "next_state" : next_state,
            "done" : done,
        }
        self.memory.append(e)
    
    def enougth_samples(self):
        return len(self.memory) >= self.BATCH_SIZE

    def sample(self):        
        # if len(self.memory) <= self.BATCH_SIZE:
        #     return None

        experiences = random.sample(self.memory, k=self.BATCH_SIZE)
        
        return experiences
    