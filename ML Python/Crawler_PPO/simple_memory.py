import numpy as np
import random

class SimpleMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, keys): 
        self.KEYS = keys

        self._reset_memory()

    def _reset_memory(self):
        self.memory = {}

        for key in self.KEYS:
            self.memory[key] = []
    
    def add(self, key, state, action, reward, next_state, done):
        """Add a new experience to memory."""            
        e = {
            "state" : state,
            "action" : action,
            "reward" : reward,
            "next_state" : next_state,
            "done" : done,
        }
        self.memory[key].append(e)
    
    def sample(self):        
        temp = self.memory.copy()

        self._reset_memory()
        
        return temp
    