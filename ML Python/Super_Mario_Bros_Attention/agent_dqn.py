import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent:

    def __init__(
        self, 
        device,
        seq_len,
        action_size,        
        eps, eps_decay, eps_min,
        burnin, update_every, batch_size, gamma, tau,
        attention_model, attention_target, dqn_model, dqn_target,
        optimizer,
        buffer_size,
        checkpoint_attention, checkpoint_dqn
        ):

        self.DEVICE = device

        # HYPERPARAMETERS           
        self.SEQ_LEN = seq_len
        self.ACTION_SIZE = action_size
        self.EPS = eps
        self.EPS_DECAY = eps_decay
        self.EPS_MIN = eps_min

        self.BURNIN = burnin
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau

        # NEURAL MODEL
        self.attention_model = attention_model
        self.attention_target = attention_target        
        self.dqn_model = dqn_model
        self.dqn_target = dqn_target
        
        self.optimizer = optimizer
        self.scaler = GradScaler() 

        self.CHECKPOINT_ATTENTION = checkpoint_attention        
        self.CHECKPOINT_DQN = checkpoint_dqn

        # MEMORY
        self.memory = MemoryBuffer(buffer_size)

        # AUX        
        self.t_step = 0
        self.l_step = 0
                
        self.loss = (0.0, 1.0e-10)        
    
    def act(self, state):

        action = None
        dist = None
        if np.random.uniform() < self.EPS:
            action_values = np.random.uniform( -1, 1, self.ACTION_SIZE ).reshape(1, self.ACTION_SIZE)

        else:            
            state = torch.tensor(state).unsqueeze(0).float().to(self.DEVICE)            
            
            self.attention_model.eval()
            self.dqn_model.eval()

            with torch.no_grad():            
                encoded = self.attention_model(state)[:,-1:].squeeze(1)
                action_values = self.dqn_model(encoded)
                        
            self.attention_model.train()
            self.dqn_model.train()
                    
            action_values = action_values.cpu().data.numpy()            
        
        dist = action_values
        action = np.argmax( action_values )

        self.EPS *= self.EPS_DECAY
        self.EPS = max(self.EPS_MIN, self.EPS)        
            
        return dist, action

    def step(self, state, dist, action, reward, next_state, done):
        self.memory.add( state, dist, action, reward, next_state, done )        
        
        # Increment step
        self.t_step += 1

        if self.t_step < self.BURNIN:
            return self.loss[0]/self.loss[1]

        # Learn every UPDATE_EVERY time steps.
        self.l_step = (self.l_step + 1) % self.UPDATE_EVERY

        if self.l_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                loss = self._learn()
                
                self.loss = (self.loss[0] * 0.99 + loss, self.loss[1] * 0.99 + 1.0)

        return self.loss[0]/self.loss[1]

    def _learn(self):        
        states, dists, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

        # TEMPORAL CORRELATION BETWEEN REWARDS

        discount = 0.9**np.arange( self.SEQ_LEN )
        rewards_future = rewards * discount
        rewards_future = rewards_future[::-1].cumsum(axis=1)[::-1]

        # TENSORS

        states         = torch.from_numpy( states                 ).float().to(self.DEVICE)
        dists          = torch.from_numpy( dists                  ).float().to(self.DEVICE).squeeze(2)
        actions        = torch.from_numpy( actions                ).long().to(self.DEVICE)
        rewards        = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        rewards_future = torch.from_numpy( rewards_future.copy()  ).float().to(self.DEVICE)
        next_states    = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones          = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE)

        # DQN / GPT-2
        with autocast(): 
            with torch.no_grad():
                encoded_ns = self.attention_target(next_states)
                Q_target_next = self.dqn_target(encoded_ns)
                Q_target_next = Q_target_next.max(2)[0]
                
                Q_target = rewards + self.GAMMA * Q_target_next * (1 - dones)
            
            encoded_s = self.attention_model(states)
            Q_value = self.dqn_model(encoded_s)
            Q_value = Q_value.gather(2, actions.unsqueeze(1)).squeeze(1)            

            loss = F.mse_loss(Q_value, Q_target)

            l2_factor = 1e-8

            l2_reg_dqn = None
            for W in self.dqn_model.parameters():
                if l2_reg_dqn is None:
                    l2_reg_dqn = W.norm(2)
                else:
                    l2_reg_dqn = l2_reg_dqn + W.norm(2)

            l2_reg_dqn = l2_reg_dqn * l2_factor

            loss += l2_reg_dqn

            l2_reg_attention = None
            for W in self.attention_model.parameters():
                if l2_reg_attention is None:
                    l2_reg_attention = W.norm(2)
                else:
                    l2_reg_attention = l2_reg_attention + W.norm(2)

            l2_reg_attention = l2_reg_attention * l2_factor

            loss += l2_reg_attention

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward() 
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._soft_update_target_model()              

        return loss.cpu().data.numpy().item()

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.dqn_target.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

        for target_param, model_param in zip(self.attention_target.parameters(), self.attention_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.attention_model.checkpoint(self.CHECKPOINT_ATTENTION)        
        self.dqn_model.checkpoint(self.CHECKPOINT_DQN)
