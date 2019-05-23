import numpy as np
from collections import deque
from collections import namedtuple
import random

class Memory:        
    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple('Experience', 
            field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""        
        e = self.experience( state, action, reward, next_state, done )
        self.memory.append(e)

    def sample(self):
        experiences = random.sample( self.memory, k=self.batch_size )

        states = np.stack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.stack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])
                
        return states, actions, rewards, next_states, dones

    def __len__(self):    
        return len(self.memory)
