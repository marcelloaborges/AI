import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Optimizer:

    def __init__(
        self, 
        device,
        memory,        
        model, target_model, optimizer,
        alpha, gamma, TAU, update_every, buffer_size, batch_size, LR,
        ):

        self.DEVICE = device
        
        # MEMORY
        self.memory = memory

        # NEURAL MODEL
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer


        # HYPERPARAMETERS
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.TAU = TAU
        self.UPDATE_EVERY = update_every
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.LR = LR

        self.t_step = 0 
        self.loss = 0      

    def step(self, state, hidden, action, reward, next_state, next_hidden, done):
        self.memory.add( state.T, hidden, action, reward, next_state.T, next_hidden, done )

         # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory then learn            
            if self.memory.enougth_samples():
                self.loss = self._learn()

        return self.loss

        
    def _learn(self):                    
        states, hiddens, actions, rewards, next_states, next_hiddens, dones, importance, sample_indices = self.memory.sample()


        states       = torch.from_numpy( states                 ).float().to(self.DEVICE)           
        actions      = torch.from_numpy( actions                ).long().to(self.DEVICE) # .squeeze(1)        
        rewards      = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        next_states  = torch.from_numpy( next_states            ).float().to(self.DEVICE)        
        dones        = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE) # .squeeze(1)       
        importance   = torch.from_numpy( importance             ).float().to(self.DEVICE) 

        hiddens      = [ tuple( [ torch.from_numpy( hc ).float().to(self.DEVICE)  for hc in hidden       ] ) for hidden in hiddens ]
        next_hiddens = [ tuple( [ torch.from_numpy( nhc ).float().to(self.DEVICE) for nhc in next_hidden ] ) for next_hidden in next_hiddens ]

        losses = []
        for i in range(len(states)):

            Q_targets_next, _ = self.target_model(next_states[i].unsqueeze(0), next_hiddens[i])
            Q_targets_next = Q_targets_next.detach().max(2)[0].squeeze(0)
            Q_target = self.ALPHA * (rewards[i] + self.GAMMA * Q_targets_next * (1 - dones[i]))            

            Q_value, _ = self.model(states[i].unsqueeze(0), hiddens[i])
            Q_value = Q_value.squeeze(0).squeeze(0).gather(0, actions.unsqueeze(1)[i])
                    
            # loss = F.smooth_l1_loss(Q_value, Q_target)
            loss = Q_value - Q_target
            loss = ( loss ** 2 ) * importance[i]
            losses.append(loss)
        
        q_loss = torch.stack( losses ).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # update target model
        self.soft_update_target_model()

        # update memory priorities
        self.memory.set_priorities(sample_indices, loss.cpu().data.numpy())

        return q_loss.data

    def soft_update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)