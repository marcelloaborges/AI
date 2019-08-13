import numpy as np
from collections import deque
import random

class Memory:        
    def __init__(self):

        self.memory = []

    def add(self, state, action, log_prob, reward):
        """Add a new experience to memory."""
        e = {
            "state" : state,
            "action" : action,
            "log_prob" : log_prob,
            "reward" : reward
        }

        self.memory.append(e)

    def experiences(self, clear=True):        
        states    = []
        actions   = []
        log_probs = []
        rewards   = []

        for exp in self.memory:                        
            states.append   ( exp['state']    )           
            actions.append  ( exp['action']   )
            log_probs.append( exp['log_prob'] )
            rewards.append  ( exp['reward']   )

        states    = np.array(states)
        actions   = np.array(actions)
        log_probs = np.array(log_probs)
        rewards   = np.array(rewards)

        n_exp = len(self)

        if clear:
            self.clear()

        return states, actions, log_probs, rewards, n_exp

    def clear(self):
        self.memory.clear()
    
    def __len__(self):    
        return len(self.memory)

    
