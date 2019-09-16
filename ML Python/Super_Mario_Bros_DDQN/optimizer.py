import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class Optimizer:

    def __init__(
        self, 
        device,
        memory,
        encoder, decoder, ddqn_model, ddqn_target, icm_target, icm,
        vae_optimizer, ddqn_optimizer, icm_optimizer,
        action_size, batch_size,
        vae_samples, compressed_features_size,
        alpha, gamma, TAU, update_every
        ):

        self.DEVICE = device
        
        # MEMORY
        self.memory = memory

        # NEURAL MODEL
        self.encoder = encoder
        self.decoder = decoder

        self.ddqn_model = ddqn_model
        self.ddqn_target = ddqn_target

        self.icm_target = icm_target
        self.icm = icm

        self.vae_optimizer = vae_optimizer
        self.ddqn_optimizer = ddqn_optimizer
        self.icm_optimizer = icm_optimizer

        # HYPERPARAMETERS
        self.ACTION_SIZE = action_size
        self.BATCH_SIZE = batch_size     

        self.VAE_SAMPLES = vae_samples
        self.COMPRESSED_FEATURES_SIZE = compressed_features_size

        self.ALPHA = alpha
        self.GAMMA = gamma
        self.TAU = TAU
        self.UPDATE_EVERY = update_every        


        self.t_step = 0
        self.icm_loss = 0
        self.vae_loss = 0
        self.ddqn_loss = 0   
        self.encoder_check = []     

    def step(self, state, action, reward, next_state, done):        
        self.memory.add( state, action, reward, next_state, done )

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory then learn            
            if self.memory.enougth_samples():
                self.vae_loss, self.ddqn_loss, self.icm_loss, self.encoder_check = self._learn()

        return self.icm_loss, self.vae_loss, self.ddqn_loss, self.encoder_check

    def _reparameterize(self, mu, logvar, samples=1):        
        if samples == 1:
            return self._z(mu, logvar)
        else:
            samples_z = []

            for _ in range(samples):
                samples_z.append( self._z(mu, logvar) )    

            return samples_z

    def _z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(mu)

        return z
        
    def _learn(self):                    
        states, actions, rewards, next_states, dones = self.memory.sample_inverse_dist()
        # states, actions, rewards, next_states, dones = self.memory.sample()

        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)                              
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE).squeeze(0)        
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE).squeeze(0)  
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)                
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE).squeeze(0)

        # ICM
        self.icm_optimizer.zero_grad()

        # intrinsic reward               
        with torch.no_grad():
            mu_states, logvar_states = self.encoder(states)
            encoded_states = self._reparameterize(mu_states, logvar_states)

            mu_next_states, logvar_next_states = self.encoder(next_states)
            encoded_next_states = self._reparameterize(mu_next_states, logvar_next_states)
            action_values = self.ddqn_model(encoded_states)

        icm_target_features = self.icm_target(encoded_next_states)
        icm_features, inverse_actions = self.icm( encoded_states, action_values, icm_target_features )

        Ri = ( torch.sum( ( icm_target_features - icm_features ).pow(2), dim=1 ) ) / 2        
        # inverse_model_loss = ( torch.sum( ( action_values - inverse_actions ).pow(2), dim=1 ) ) / 2
        
        ce = nn.CrossEntropyLoss()        
        picked_actions = action_values.max(1)[1]
        
        inverse_model_loss = ce( inverse_actions, picked_actions )

        # Loss
        icm_loss = Ri.mean() + inverse_model_loss.mean()
        
        icm_loss.backward()
        nn.utils.clip_grad_norm_( self.icm.parameters(), 0.5 )
        self.icm_optimizer.step()

        # VAE

        self.vae_optimizer.zero_grad()

        mu_states, logvar_states = self.encoder(states)
        encoded_states = self._reparameterize( mu_states, logvar_states, self.VAE_SAMPLES )

        decoded_states = [ self.decoder( z ) for z in encoded_states ]        

        # MSE
        MSE = 0
        for recon_x in decoded_states:
            exp = ( states.reshape((-1, 3, 240 * 256)) - recon_x.reshape((-1, 3, 240 * 256)) ) ** 2
            MSE += ( ( exp ).sum(dim=2) ).mean()
        MSE /= self.VAE_SAMPLES * self.BATCH_SIZE

        # KLD
        KLD = -0.5 * torch.sum(1 + logvar_states - mu_states.pow(2) - logvar_states.exp())
        KLD /= self.BATCH_SIZE * self.COMPRESSED_FEATURES_SIZE

        # L2 regularization
        l2_factor = 1e-6
        
        l2_encoder_reg = None
        for W in self.encoder.parameters():
            if l2_encoder_reg is None:
                l2_encoder_reg = W.norm(2)
            else:
                l2_encoder_reg = l2_encoder_reg + W.norm(2)

        l2_encoder_reg = l2_encoder_reg * l2_factor

        l2_decoder_reg = None
        for W in self.decoder.parameters():
            if l2_decoder_reg is None:
                l2_decoder_reg = W.norm(2)
            else:
                l2_decoder_reg = l2_decoder_reg + W.norm(2)

        l2_decoder_reg = l2_decoder_reg * l2_factor

        vae_loss = MSE + KLD + l2_encoder_reg + l2_decoder_reg


        vae_loss.backward()
        self.vae_optimizer.step()


        # Q Loss

        self.ddqn_optimizer.zero_grad()

        mu_next_states, logvar_next_states = self.encoder(next_states)
        encoded_next_states = self._reparameterize( mu_next_states, logvar_next_states )

        with torch.no_grad():
            Q_target_next = self.ddqn_target(encoded_next_states).max(1)[0]

            # ie_rewards = torch.clamp( rewards + ( Ri.detach() * 0.0001 ), -15, 1 )
            ie_rewards = rewards + Ri.detach() * 0.05
            Q_target = self.ALPHA * (ie_rewards + self.GAMMA * Q_target_next * (1 - dones))

        mu_states, logvar_states = self.encoder(states)
        encoded_states = self._reparameterize(mu_states, logvar_states)
        Q_value = self.ddqn_model(encoded_states).gather(1, actions.unsqueeze(1)).squeeze(1)

        q_loss = F.mse_loss(Q_value, Q_target)
        # q_loss = ( (Q_value - Q_target) ** 2).mean()

        # L2 Regularization      
        
        l2_factor = 0.0005
        l2_reg = None
        for W in self.ddqn_model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        l2_reg = l2_reg * l2_factor

        # # Entropy regularization

        # # tf.reduce_sum( tf.nn.softmax( q ) * tf.log( tf.nn.softmax( q ) + 1e-10 ), axis = 1 )    
        # entropy_factor = 0.01    
        # entropy = - ( torch.softmax( Q_value, dim=0 ) * torch.log( torch.softmax( Q_value, dim=0 ) + 1e-10 ) ).mean()
        # entropy = entropy * entropy_factor

        # Loss L2 and entropy applied
        q_loss = q_loss + l2_reg # + entropy
        
        q_loss.backward()
        # nn.utils.clip_grad_norm_( self.model.parameters(), 1 )
        # nn.utils.clip_grad_norm_( self.cnn.parameters(), 1 )

        self.ddqn_optimizer.step()

        # update target model
        self.soft_update_target_model()
       
        return vae_loss.item(), icm_loss.item(), q_loss.item(), decoded_states

    def soft_update_target_model(self):
        for target_param, model_param in zip(self.ddqn_target.parameters(), self.ddqn_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)