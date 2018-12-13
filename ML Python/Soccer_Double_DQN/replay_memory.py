import torch
import numpy as np
import random
from collections import deque, namedtuple

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""        
        
        self.memory = deque(maxlen=buffer_size)
        self.BATCH_SIZE = batch_size        
        self.experience = namedtuple('Experience', ["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None

        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.BATCH_SIZE)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)