import numpy as np
from collections import namedtuple
import random

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, keys):
        """Initialize a ReplayMemory object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """                
        # self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        # self.batch_size = batch_size
        
        self.memory = {}

        self.experience = namedtuple('Experience', field_names=["states", "actions", "rewards", "next_states", "dones"])

        for key in keys:
            self.memory[key] = []
    
    def add(self, key, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""    
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory[key].append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""        
        experiences = random.sample(self.memory[self.KEYS], k=self.batch_size)

        keys        = np.vstack( [ e.keys        for e in experiences if e is not None ] )
        states      = np.vstack( [ e.states      for e in experiences if e is not None ] )
        actions     = np.vstack( [ e.actions     for e in experiences if e is not None ] )
        rewards     = np.vstack( [ e.rewards     for e in experiences if e is not None ] )
        next_states = np.vstack( [ e.next_states for e in experiences if e is not None ] )
        dones       = np.vstack( [ e.dones       for e in experiences if e is not None ] )

        return keys, states, actions, rewards, next_states, dones

    def enough_experiences(self):
        return len(self) >= self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len( self.memory[self.KEYS] )

