import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory_buffer import MemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent:

    def __init__(
        self, 
        device,
        action_size,        
        update_every, batch_size, gamma, tau,
        attention_model, model, target_model,
        optimizer,        
        checkpoint_dqn
        ):

        self.DEVICE = device

        # HYPERPARAMETERS           
        self.ACTION_SIZE = action_size
                                
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau

        # NEURAL MODEL
        self.attention_model = attention_model
        self.model = model
        self.target_model = target_model

        self.attention_model.eval()

        self.optimizer = optimizer

        self.CHECKPOINT_DQN = checkpoint_dqn

        # MEMORY
        self.memory = MemoryBuffer()

        # AUX        
        self.t_step = 0
        self.loss = 0
    
    def act(self, state, eps=0.):        
        state = torch.tensor(state).unsqueeze(0).float().to(self.DEVICE)
        
        self.model.eval()

        with torch.no_grad():            
            encoded = self.attention_model(state)
            action_values = self.model(encoded[:,-1:].squeeze(1))
                    
        self.model.train()
                
        action_values = action_values.cpu().data.numpy()
        
        action = None
        if np.random.uniform() < eps:
            action = random.choice( np.arange(self.ACTION_SIZE) )
        else:
            action = np.argmax( action_values )
            
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add( state, action, reward, next_state, done )
            
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY

        if self.t_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                self.loss = self._learn()

        return self.loss

    def _learn(self):        
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)


        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE).squeeze(0)        
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE).squeeze(0)  
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE).squeeze(0)
        

        # DQN
        
        with torch.no_grad():
            encoded_ns = self.attention_model(next_states)[:,-1:].squeeze(1)
            Q_target_next = self.target_model(encoded_ns)
            Q_target_next = Q_target_next.max(1)[0]
            
            Q_target = rewards + self.GAMMA * Q_target_next * (1 - dones)

        encoded_s = self.attention_model(states)[:,-1:].squeeze(1)
        Q_value = self.model(encoded_s)
        Q_value = Q_value.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(Q_value, Q_target)


        l2_factor = 1e-8

        l2_reg_actor = None
        for W in self.model.parameters():
            if l2_reg_actor is None:
                l2_reg_actor = W.norm(2)
            else:
                l2_reg_actor = l2_reg_actor + W.norm(2)

        l2_reg_actor = l2_reg_actor * l2_factor

        loss += l2_reg_actor


        self.optimizer.zero_grad()                  
        loss.backward()        
        self.optimizer.step()        

        return loss.cpu().data.numpy().item()
   

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.model.checkpoint(self.CHECKPOINT_DQN)
