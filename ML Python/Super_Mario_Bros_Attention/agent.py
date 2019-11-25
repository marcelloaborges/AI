import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Model

# from unusual_memory import UnusualMemory
from memory import Memory

class Agent:

    def __init__(
        self, 
        device,
        channels,
        compressed_features_size,
        action_size,
        lr,
        buffer_size, 
        update_every, batch_size,
        vae_samples,
        img_w, img_h,
        gamma, tau,
        checkpoint_folder='./'
        ):

        self.DEVICE = device

        self.CHANNELS = channels
        self.ACTION_SIZE = action_size        

        # NEURAL MODEL
        self.model = Model(channels, compressed_features_size, 256, vae_samples, action_size, 32)
        self.target_model = Model(channels, compressed_features_size, 256, vae_samples, action_size, 32)

        self.optimizer = optim.Adam( self.model.parameters(), lr=lr )

        # MEMORY
        self.memory = Memory(buffer_size)

        # HYPERPARAMETERS        
        self.COMPRESSED_FEATURES_SIZE = compressed_features_size
        self.VAE_SAMPLES = vae_samples        
        self.img_w = img_w
        self.img_h = img_h

        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau        

        # LOAD CHECKPOING
        self.CHECKPOINT_MODEL = checkpoint_folder + 'CHECKPOINT.pth'        

        self.model.load(self.CHECKPOINT_MODEL, self.DEVICE)
        self.target_model.load(self.CHECKPOINT_MODEL, self.DEVICE)

        # AUX
        self.vae_loss = 0
        self.q_loss = 0        
        self.encoder_check = []

        self.t_step = 0        
        
    def _reparameterize(self, mu, logvar, samples=4):
        samples_z = []

        for _ in range(samples):
            samples_z.append( self._z(mu, logvar) )    

        return samples_z

    def _z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)

        return z
    
    def act(self, state, eps=0.):                
        state = torch.tensor(state).float().unsqueeze(0).to(self.DEVICE)

        self.model.eval()        

        with torch.no_grad():            
            action_values, _, _, _ = self.model(state, False)

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

        # If enough samples are available in memory then learn
        if self.t_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                self.vae_loss, self.q_loss, self.encoder_check = self._learn()

        return self.vae_loss, self.q_loss, self.encoder_check

    def _learn(self):
        # states, actions, rewards, next_states, dones = self.memory.sample_inverse_dist(self.VAE_BATCH_SIZE)
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE).squeeze(0)        
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE).squeeze(0)  
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE).squeeze(0)


        self.optimizer.zero_grad()

        Q_value, decoded_states, mu_states, logvar_states = self.model(states)

        # VAE                        

        # MSE
        MSE = 0
        for recon_x in decoded_states:
            exp = ( 
                states.reshape((-1, self.CHANNELS, self.img_h * self.img_w)) - 
                recon_x.reshape((-1, self.CHANNELS, self.img_h * self.img_w)) 
                ) ** 2
            MSE += ( ( exp ).sum(dim=2) ).mean()
        MSE /= self.VAE_SAMPLES * self.BATCH_SIZE

        # KLD
        KLD = -0.5 * torch.sum(1 + logvar_states - mu_states.pow(2) - logvar_states.exp())
        KLD /= self.BATCH_SIZE * self.COMPRESSED_FEATURES_SIZE        

        vae_loss = MSE + KLD


        # DQN

        with torch.no_grad():
            Q_target_next, _, _, _ = self.target_model(next_states)
            Q_target_next = Q_target_next.max(1)[0]
            
            Q_target = self.GAMMA * Q_target_next * (1 - dones)
        
        Q_value = Q_value.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_loss = F.mse_loss(Q_value, Q_target)

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
        loss = vae_loss + q_loss + l2_reg

        loss.backward()
        self.optimizer.step()        

        # update target model        
        # if self.target_update_step == 0:       
        #     self._update_target_model()
        self._soft_update_target_model()

        return vae_loss.item(), q_loss.item(), decoded_states

    def _update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(model_param.data)

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.model.checkpoint(self.CHECKPOINT_MODEL)