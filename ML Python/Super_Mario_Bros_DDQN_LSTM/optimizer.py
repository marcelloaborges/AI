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

    def step(self, state, hx, cx, action, reward, next_state, nhx, ncx, done):
        self.memory.add( 
            state.T, hx.squeeze(0), cx.squeeze(0), action, reward, next_state.T, nhx.squeeze(0), ncx.squeeze(0), done 
            )

         # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory then learn            
            if self.memory.enougth_samples():
                self.loss = self._learn()

        return self.loss

        
    def _learn(self):                    
        states, hxs, cxs, actions, rewards, next_states, nhxs, ncxs, dones, importance, sample_indices = self.memory.sample()


        states       = torch.from_numpy( states                 ).float().to(self.DEVICE)           
        actions      = torch.from_numpy( actions                ).long().to(self.DEVICE) # .squeeze(1)        
        rewards      = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        next_states  = torch.from_numpy( next_states            ).float().to(self.DEVICE)        
        dones        = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE) # .squeeze(1)       
        importance   = torch.from_numpy( importance             ).float().to(self.DEVICE) 

        hxs          = torch.from_numpy( hxs                    ).float().to(self.DEVICE)
        cxs          = torch.from_numpy( cxs                    ).float().to(self.DEVICE)
        nhxs         = torch.from_numpy( nhxs                   ).float().to(self.DEVICE)
        ncxs         = torch.from_numpy( ncxs                   ).float().to(self.DEVICE)

        Q_targets_next, _, _ = self.target_model(next_states, nhxs, ncxs)
        Q_targets_next = Q_targets_next.detach().max(2)[0].squeeze()
        Q_target = self.ALPHA * (rewards + self.GAMMA * Q_targets_next * (1 - dones))

        Q_value, _, _ = self.model(states, hxs, cxs)
        Q_value = Q_value.squeeze().gather(1, actions.unsqueeze(1))
                
        # loss = F.smooth_l1_loss(Q_value, Q_target)
        loss = Q_value.squeeze() - Q_target
        loss = ( loss ** 2 ) * importance
        q_loss = loss.mean()

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