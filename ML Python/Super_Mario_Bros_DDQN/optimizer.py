import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class Optimizer:

    def __init__(
        self, 
        device,
        memory,
        cnn, model, target, optimizer,
        rnd_target, rnd_predictor, rnd_optimizer,
        alpha, gamma, TAU, update_every, buffer_size, batch_size
        ):

        self.DEVICE = device
        
        # MEMORY
        self.memory = memory

        # NEURAL MODEL
        self.cnn = cnn
        self.model = model
        self.target = target
        self.optimizer = optimizer

        # RND
        self.rnd_target = rnd_target
        self.rnd_predictor = rnd_predictor        
        self.rnd_optimizer = rnd_optimizer

        # HYPERPARAMETERS
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.TAU = TAU
        self.UPDATE_EVERY = update_every
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size        

        
        self.t_step = 0 
        self.loss = 0
        self.rnd_loss = 0

    def step(self, state, action, reward, next_state, done):        
        self.memory.add( np.stack(state), action, reward, np.stack(next_state), done )        

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory then learn            
            if self.memory.enougth_samples():
                self.loss, self.rnd_loss = self._learn()

        return self.loss, self.rnd_loss

        
    def _learn(self):                    

        states, actions, rewards, next_states, dones = self.memory.sample_inverse_dist()
        
        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)           
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE).squeeze(0)        
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE).squeeze(0)  
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)        
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE).squeeze(0)

        # RND
        self.rnd_optimizer.zero_grad()

        # RND intrinsic reward               
        with torch.no_grad():
            next_states_flatten = self.cnn(next_states)
            rnd_target = self.rnd_target(next_states_flatten)

        rnd_predictor = self.rnd_predictor( next_states_flatten )

        Ri = ( torch.sum( ( rnd_target - rnd_predictor ).pow(2), dim=1 ) ) / 2

        # RND Loss
        rnd_loss = Ri.mean()
        
        rnd_loss.backward()
        nn.utils.clip_grad_norm_( self.rnd_predictor.parameters(), 0.5 )

        self.rnd_optimizer.step()

        # Q Loss

        self.optimizer.zero_grad()

        next_states_flatten = self.cnn(next_states)

        with torch.no_grad():
            Q_target_next = self.target(next_states_flatten).max(1)[0]                        
            # Q_target = self.ALPHA * (rewards + self.GAMMA * Q_target_next * (1 - dones))

            # ie_rewards = torch.clamp( rewards + ( Ri.detach() * 0.0001 ), -15, 1 )
            ie_rewards = rewards + Ri.detach() * 0.0005
            Q_target = self.ALPHA * (ie_rewards + self.GAMMA * Q_target_next * (1 - dones))

        states_flatten = self.cnn(states)
        Q_value = self.model(states_flatten).gather(1, actions.unsqueeze(1)).squeeze(1)

        q_loss = F.smooth_l1_loss(Q_value, Q_target)
        # loss = Q_value - Q_target
        # loss = ( loss ** 2 ) * importances
        # q_loss = loss.mean()

        # L2 Regularization      
        
        l2_factor = 0.0005
        l2_reg = None
        for W in self.model.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)

        l2_reg = l2_reg * l2_factor

        # Entropy regularization

        # tf.reduce_sum( tf.nn.softmax( q ) * tf.log( tf.nn.softmax( q ) + 1e-10 ), axis = 1 )    
        entropy_factor = 0.01    
        entropy = - ( torch.softmax( Q_value, dim=0 ) * torch.log( torch.softmax( Q_value, dim=0 ) + 1e-10 ) ).mean()
        entropy = entropy * entropy_factor

        # Loss L2 and entropy applied
        f_loss = q_loss + l2_reg + entropy
        
        f_loss.backward()
        # nn.utils.clip_grad_norm_( self.model.parameters(), 1 )
        # nn.utils.clip_grad_norm_( self.cnn.parameters(), 1 )

        self.optimizer.step()

        # update target model
        self.soft_update_target_model()
       
        return f_loss.item(), rnd_loss.item()

    def soft_update_target_model(self):
        for target_param, model_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)