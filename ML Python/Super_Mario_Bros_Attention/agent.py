import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Attention

from unusual_memory import UnusualMemory
# from memory import Memory

class Agent:

    def __init__(
        self, 
        device,
        channels,        
        action_size,
        lr,
        buffer_size, 
        frame_skip, update_every, batch_size,                
        gamma, tau,
        checkpoint_folder='./'
        ):

        self.DEVICE = device

        # HYPERPARAMETERS
        self.CHANNELS = channels
        self.ACTION_SIZE = action_size   
                   
        self.FRAME_SKIP = frame_skip
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau             

        # NEURAL MODEL
        self.model = Attention(action_size, channels, 128, 64)
        self.target_model = Attention(action_size, channels, 128, 64)

        self.optimizer = optim.Adam( self.model.parameters(), lr=lr )

        # MEMORY
        self.memory = UnusualMemory(buffer_size)        
        # self.memory = Memory(buffer_size)        

        # LOAD CHECKPOING
        self.CHECKPOINT_MODEL = checkpoint_folder + 'CHECKPOINT.pth'

        self.model.load(self.CHECKPOINT_MODEL, self.DEVICE)
        self.target_model.load(self.CHECKPOINT_MODEL, self.DEVICE)

        # AUX
        self.loss = 0
        self.t_frame_skip = 0
        self.t_step = 0                
    
    def act(self, state, eps=0.):                
        state = torch.tensor(state).float().unsqueeze(0).to(self.DEVICE)

        self.model.eval()        

        with torch.no_grad():            
            action_values = self.model(state)

        self.model.train()

        action_values = action_values.cpu().data.numpy()
        
        action = None
        if np.random.uniform() < eps:
            action = random.choice( np.arange(self.ACTION_SIZE) )
        else:
            action = np.argmax( action_values )
            
        return action

    def step(self, state, action, reward, next_state, done):
        self.t_frame_skip = (self.t_frame_skip + 1) % self.FRAME_SKIP

        if self.t_frame_skip == 0:
            self.memory.add( state, action, reward, next_state, done )
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY

        # If enough samples are available in memory then learn
        if self.t_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                self.loss = self._learn()

        return self.loss

    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample_inverse_dist(self.BATCH_SIZE)
        # states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE).squeeze(0)        
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE).squeeze(0)  
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE).squeeze(0)
        

        # DQN

        with torch.no_grad():
            Q_target_next = self.target_model(next_states)
            Q_target_next = Q_target_next.max(1)[0]
            
            Q_target = self.GAMMA * Q_target_next * (1 - dones)
        
        Q_value = self.model(states)
        Q_value = Q_value.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_loss = F.smooth_l1_loss(Q_value, Q_target)

        # L2 Regularization              
        l2_factor = 1e-6

        l2_reg = None        
        for W in self.model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        l2_reg = l2_reg * l2_factor        

        # Loss
        loss = q_loss + l2_reg


        # Apply gradients
        self.optimizer.zero_grad()        
        loss.backward()
        self.optimizer.step()        

        # update target model        
        # if self.target_update_step == 0:       
        #     self._update_target_model()
        self._soft_update_target_model()

        return q_loss.item()

    def _update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(model_param.data)

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.model.checkpoint(self.CHECKPOINT_MODEL)