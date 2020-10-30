import numpy as np

class SimpleMemory:        
    def __init__(self):                
        self.memory = []

    def add(self, state, action, reward, next_state, done):
        """Add a new sequence to memory."""
        e = {
            "state" : state,
            "action" : action,
            "reward" : reward,
            "next_state" : next_state,
            "done" : done
        }

        self.memory.append(e)    

    def experiences(self):        

        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        for exp in self.memory:                        
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

        self.clear()

        return states, actions, rewards, next_states, dones

    def clear(self):
        self.memory = []
