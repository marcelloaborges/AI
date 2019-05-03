import numpy as np
import random
import os

import torch
import torch.optim as optim

from replay_memory import ReplayMemory

class Agent:

    def __init__(self, 
        device,        
        actor_model,
        local_memory,
        shared_memory,
        noise,
        gamma,
        n_step):

        self.DEVICE = device

        # Actor Network
        self.actor_model = actor_model

        # Replay memory        
        self.local_memory = local_memory
        self.shared_memory = shared_memory

        # Noise process
        self.noise = noise        

        # Hyperparameters
        self.GAMMA = gamma
        self.N_STEP = n_step    

        self.t_step = 0 
        
    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.actor_model.eval()
        with torch.no_grad():
            action = self.actor_model(state).cpu().data.numpy()
        self.actor_model.train()        

        if add_noise:
            action += self.noise.sample()
    
        return np.clip(action, -1, 1)

    def step(self, keys, state, action, reward, next_state, done):
        # Save experience / reward        
        for i, key in enumerate(keys):
            self.local_memory.add( 
                key,
                state[i],
                action[i],
                np.array( reward[i] ),
                next_state[i],
                np.array( done[i] )
            )

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:            
            experiences = self.local_memory.sample()                    

            for key, experience in experiences.items():

                states      = []
                actions     = []
                rewards     = []
                next_states = []
                dones       = []

                temp_reward = []
                for exp in experience:
                    states.append(      exp['state']      )
                    actions.append(     exp['action']     )
                    temp_reward.append( exp['reward']     )
                    next_states.append( exp['next_state'] )
                    dones.append(       exp['done']       )

                discount = self.GAMMA**np.arange(len(temp_reward))
                temp_reward = temp_reward * discount
                temp_reward = temp_reward[::-1].cumsum(axis=0)[::-1]

                rewards.extend( temp_reward )
                
                for i in range(len(states)):
                    self.shared_memory.add(                        
                        states[i],
                        actions[i],
                        rewards[i],
                        next_states[i],
                        dones[i]
                    )

    def reset(self):
        self.noise.reset()