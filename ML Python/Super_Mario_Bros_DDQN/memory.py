import numpy as np
from collections import deque
import random

class Memory:        
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
            "done" : done
        }

        self.memory.append(e)

    def _enougth_samples(self):
        return len( self.memory ) >= self.BATCH_SIZE

    def sample(self):        
        if not self._enougth_samples():            
            return None
        
        samples = random.sample( self.memory, k=self.BATCH_SIZE )        

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

    
