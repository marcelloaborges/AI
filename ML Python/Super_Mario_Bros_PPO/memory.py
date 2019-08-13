import numpy as np
from collections import deque
import random

class Memory:        
    def __init__(self):

        self.memory = []

    def add(self, state, action, log_prob, reward, next_state):
        """Add a new experience to memory."""
        e = {
            "state" : state,
            "action" : action,
            "log_prob" : log_prob,
            "reward" : reward,
            "next_state" : next_state
        }

        self.memory.append(e)

    def experiences(self, clear=True):        
        states      = []
        actions     = []
        log_probs   = []
        rewards     = []
        next_states = []

        for exp in self.memory:                        
            states.append     ( exp['state']      )           
            actions.append    ( exp['action']     )
            log_probs.append  ( exp['log_prob']   )
            rewards.append    ( exp['reward']     )
            next_states.append( exp['next_state'] )

        states      = np.array(states)
        actions     = np.array(actions)
        log_probs   = np.array(log_probs)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)

        n_exp = len(self)

        if clear:
            self.clear()

        return states, actions, log_probs, rewards, next_states, n_exp

    def clear(self):
        self.memory.clear()
    
    def __len__(self):    
        return len(self.memory)

    
