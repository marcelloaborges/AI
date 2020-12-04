import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import wandb

class AgentDDQN:

    def __init__(
        self, 
        device,
        action_size,
        eps, eps_decay, eps_min,
        burnin, update_every, batch_size, 
        entropy_tau, alpha, lo, gamma, tau,
        dqn_model, dqn_target,
        optimizer,
        buffer_size
        ):

        self.DEVICE = device

        # HYPERPARAMETERS           
        self.ACTION_SIZE = action_size
        self.EPS = eps
        self.EPS_DECAY = eps_decay
        self.EPS_MIN = eps_min

        self.BURNIN = burnin        
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.ENTROPY_TAU = entropy_tau
        self.ALPHA = alpha
        self.LO = lo
        self.GAMMA = gamma
        self.TAU = tau        

        # NEURAL MODEL
        self.dqn_model = dqn_model
        self.dqn_target = dqn_target
        
        self.optimizer = optimizer
        self.scaler = GradScaler() 

        # MEMORY
        self.memory = MemoryBuffer(buffer_size)

        # AUX        
        self.l_step = 0
                
        # self.loss = (0.0, 1.0e-10)
        self.loss = 0
    
    def act(self, state):
        action = None
        action_values = None
        if np.random.uniform() < self.EPS:
            action_values = np.random.uniform( -3, 3, self.ACTION_SIZE ).reshape(1, self.ACTION_SIZE)

        else:            
            state = torch.from_numpy(state).float().to(self.DEVICE)
                        
            self.dqn_model.eval()

            with torch.no_grad():                            
                action_values = self.dqn_model(state)
                        
            self.dqn_model.train()
                    
            action_values = action_values.cpu().data.numpy()            
        
        action_values = action_values
        action = np.argmax( action_values )

        self.EPS *= self.EPS_DECAY
        self.EPS = max(self.EPS_MIN, self.EPS)        
            
        return action, action_values

    def step(self, state, action, reward, next_state, done):
        self.memory.add( state, action, reward, next_state, done )

        if len( self.memory ) < self.BURNIN:
            return self.loss[0]/self.loss[1]

        # Learn every UPDATE_EVERY time steps.
        self.l_step = (self.l_step + 1) % self.UPDATE_EVERY
        if self.l_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                self.loss = self._learn()                                

        return self.loss

    def _learn(self):        
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)        


        # TENSORS
        states         = torch.from_numpy( states                 ).float().to(self.DEVICE)
        actions        = torch.from_numpy( actions                ).long().to(self.DEVICE)
        rewards        = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        next_states    = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones          = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE)

        
        # DQN MUNCHAUSEN

        # with autocast(): 
        with torch.no_grad():
            # Get predicted Q values (for next states) from target model
            Q_targets_next = self.dqn_target( next_states.squeeze(1) )
            # calculate entropy term with logsum 
            logsum = torch.logsumexp( ( Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) ) / self.ENTROPY_TAU , 1 ).unsqueeze(-1)

            tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.ENTROPY_TAU * logsum
            # target policy
            pi_target = F.softmax( Q_targets_next / self.ENTROPY_TAU, dim=1 )
            Q_target = ( self.GAMMA * ( pi_target * ( Q_targets_next - tau_log_pi_next ) * ( 1 - dones ).unsqueeze(-1) ).sum(1) ).unsqueeze(-1)
            
            # calculate munchausen addon with logsum trick
            q_k_targets = self.dqn_target( states.squeeze(1) )
            v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
            logsum = torch.logsumexp( ( q_k_targets - v_k_target ) / self.ENTROPY_TAU, 1 ).unsqueeze(-1)
            log_pi = q_k_targets - v_k_target - self.ENTROPY_TAU * logsum
            munchausen_addon = log_pi.gather(1, actions.unsqueeze(-1))
            
            # calc munchausen reward:
            munchausen_reward = (rewards.unsqueeze(-1) + self.ALPHA * torch.clamp(munchausen_addon, min=self.LO, max=0))
            
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target
            
        # Get expected Q values from local model
        q_k = self.dqn_model( states.squeeze(1) )
        Q_values = q_k.gather(1, actions.unsqueeze(-1))
        
        # Compute loss
        loss = F.mse_loss(Q_values, Q_targets) # mse_loss

        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward() 
        self.scaler.step(self.optimizer)
        self.scaler.update()

        
        self._soft_update_target_model()

        return loss.cpu().data.numpy().item()

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.dqn_target.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)