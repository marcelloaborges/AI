import numpy as np
from numpy import array, mean, std

from collections import deque
import random
import pickle

class MemoryBuffer:        
    def __init__(self, buffer_size=50000):
        self.BUFFER_SIZE = buffer_size

        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, log_prob, reward, next_state, done):
        """Add a new experience to memory."""
        e = {
            "states" : state,
            "actions" : action,
            "log_probs" : log_prob,
            "rewards" : reward,
            "next_states": next_state,
            "dones": done
        }

        self.memory.append(e)
    
    def enougth_samples(self, batch_size):
        return len( self.memory ) >= batch_size

    def exp(self, clear=True):
        states      = []
        actions     = []
        log_probs   = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in self.memory:
            states.append     ( exp['states']      )           
            actions.append    ( exp['actions']     )
            log_probs.append  ( exp['log_probs']   )
            rewards.append    ( exp['rewards']     )
            next_states.append( exp['next_states'] )
            dones.append      ( exp['dones']       )

        states      = np.array(states)
        actions     = np.array(actions)
        log_probs   = np.array(log_probs)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)
        dones       = np.array(dones)

        n_exp = len(self.memory)

        if clear:
            self.memory = deque(maxlen=self.BUFFER_SIZE)

        return states, actions, log_probs, rewards, next_states, dones, n_exp

    def sample(self, batch_size):

        samples = random.sample( self.memory, k=batch_size )

        states      = []
        actions     = []
        log_probs   = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in samples:
            states.append     ( exp['states']      )           
            actions.append    ( exp['actions']     )
            log_probs.append  ( exp['log_probs']   )
            rewards.append    ( exp['rewards']     )
            next_states.append( exp['next_states'] )
            dones.append      ( exp['dones']       )

        states      = np.array(states)
        actions     = np.array(actions)
        log_probs   = np.array(log_probs)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)
        dones       = np.array(dones)

        return states, actions, log_probs, rewards, next_states, dones

    def __len__(self):    
        return len(self.memory)