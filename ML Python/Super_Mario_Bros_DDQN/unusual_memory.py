import numpy as np
from collections import deque
import random

class UnusualMemory:
    def __init__(self, buffer_size, batch_size, unusual_sample_factor=0.99):
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.UNUSUAL_SAMPLE_FACTOR = unusual_sample_factor
        
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, hx, cx, action, reward, next_state, nhx, ncx, done):
        """Add a new experience to memory."""
        e = {
            'state' : state,
            'hx' : hx,
            'cx' : cx,
            'state' : state,
            'action' : action,
            'reward' : reward,
            'next_state' : next_state,
            'nhx' : nhx,
            'ncx' : ncx,
            'done' : done
        }

        self.memory.append(e)

    def enougth_samples(self):
        return len( self.memory ) >= self.BATCH_SIZE

    def sample_abs(self):                                
        # PRIORITIZING THE UNUSUAL EXPERIENCES
        sorted_memory = sorted( self.memory, key=lambda exp: abs( exp['reward'] ), reverse=True )
        p = np.array( [ self.UNUSUAL_SAMPLE_FACTOR ** i for i in range(len(sorted_memory)) ] )
        p = p / sum(p)
        sample_idxs = random.choices( np.arange( len(sorted_memory) ), k=self.BATCH_SIZE, weights=p )
        samples = [ sorted_memory[idx] for idx in sample_idxs ] 

        states      = []
        hx          = []
        cx          = []
        actions     = []
        rewards     = []
        next_states = []
        nhx         = []
        ncx         = []
        dones       = []

        for exp in samples:                        
            states.append     ( exp['state']      )           
            hx.append         ( exp['hx']         )           
            cx.append         ( exp['cx']         )           
            actions.append    ( exp['action']     )
            rewards.append    ( exp['reward']     )
            next_states.append( exp['next_state'] )
            nhx.append        ( exp['nhx']        )           
            ncx.append        ( exp['ncx']        )           
            dones.append      ( exp['done']       )

        states      = np.array(states)
        hx          = np.array(hx)
        cx          = np.array(cx)
        actions     = np.array(actions)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)
        nhx         = np.array(nhx)
        ncx         = np.array(ncx)
        dones       = np.array(dones)

        return states, hx, cx, actions, rewards, next_states, nhx, ncx, dones

    def sample_inverse_dist(self):                         
        rewards_inverse_distribution = self._rewards_inverse_distribution()

        # PRIORITIZING UNUSUAL EXPERIENCES

        samples = []        
        rewards = [ k for k, v in rewards_inverse_distribution.items() ]
        probs = [ v for k, v in rewards_inverse_distribution.items() ]
        for _ in range( int(self.BATCH_SIZE / 2) ):
            r_chosen = random.choices( rewards, weights=probs )[0]
            reward_exp = [ exp for exp in self.memory if exp['reward'] == r_chosen ]

            samples.append( random.choice( reward_exp ) )

        samples.extend( random.sample( self.memory, k = int(self.BATCH_SIZE / 2) ) )

        states      = []
        hx          = []
        cx          = []
        actions     = []
        rewards     = []
        next_states = []
        nhx         = []
        ncx         = []
        dones       = []

        for exp in samples:                        
            states.append     ( exp['state']      )           
            hx.append         ( exp['hx']         )           
            cx.append         ( exp['cx']         )           
            actions.append    ( exp['action']     )
            rewards.append    ( exp['reward']     )
            next_states.append( exp['next_state'] )
            nhx.append        ( exp['nhx']        )           
            ncx.append        ( exp['ncx']        )           
            dones.append      ( exp['done']       )

        states      = np.array(states)
        hx          = np.array(hx)
        cx          = np.array(cx)
        actions     = np.array(actions)
        rewards     = np.array(rewards)
        next_states = np.array(next_states)
        nhx         = np.array(nhx)
        ncx         = np.array(ncx)
        dones       = np.array(dones)

        return states, hx, cx, actions, rewards, next_states, nhx, ncx, dones

    def _rewards_distribution(self):
        reward_freq = {}

        for exp in self.memory:
            if exp['reward'] in reward_freq:
                reward_freq[ exp['reward'] ] += 1
            else:
                reward_freq[ exp['reward'] ] = 1
        
        reward_dist = {}
        for k, value in reward_freq.items():
            reward_dist[k] = value / len( self.memory )

        return reward_dist

    def _rewards_inverse_distribution(self):
        reward_inverse_freq = {}

        for exp in self.memory:
            if exp['reward'] in reward_inverse_freq:
                reward_inverse_freq[ exp['reward'] ] -= 1
            else:
                reward_inverse_freq[ exp['reward'] ] = len( self.memory )
        
        total = 0
        for k, value in reward_inverse_freq.items():            
            total += value

        reward_inverse_dist = {}
        for k, value in reward_inverse_freq.items():            
            reward_inverse_dist[k] = value / total

        return reward_inverse_dist
