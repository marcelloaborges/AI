import numpy as np
from collections import deque
import random

class Memory:        
    def __init__(self, buffer_size):
        self.BUFFER_SIZE = buffer_size        
        
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = {
            "state" : state,
            "action" : action,
            "reward" : reward,
            "next_state" : next_state,
            "done" : done
        }

        self.memory.append(e)

    def enougth_samples(self, batch_size):
        return len( self.memory ) >= batch_size

    def sample(self, batch_size):
        samples = random.sample( self.memory, k=batch_size )

        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in samples:                        
            states.append     ( exp['state']      )           
            actions.append    ( exp['action']     )
            rewards.append    ( exp['reward']     )
            next_states.append( exp['next_state'] )
            dones.append      ( exp['done']       )

        states      = np.array(states)
        actions     = np.array(actions)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)
        dones       = np.array(dones)

        return states, actions, rewards, next_states, dones

    
