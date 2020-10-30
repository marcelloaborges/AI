import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer
from prioritized_memory_buffer import PrioritizedMemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Agent:

    def __init__(
        self, 
        device,
        action_size,
        eps, eps_decay, eps_min,
        burnin, n_steps, update_every, batch_size, 
        entropy_tau, alpha, lo, gamma, tau,
        cnn_model, rnd_target, rnd_predictor, dqn_model, dqn_target,
        rnd_optimizer, dqn_optimizer,
        buffer_size
        ):

        self.DEVICE = device

        # HYPERPARAMETERS
        self.ACTION_SIZE = action_size
        self.EPS = eps
        self.EPS_DECAY = eps_decay
        self.EPS_MIN = eps_min

        self.BURNIN = burnin
        self.N_STEPS = n_steps
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.ENTROPY_TAU = entropy_tau
        self.ALPHA = alpha
        self.LO = lo
        self.GAMMA = gamma
        self.TAU = tau

        # NEURAL MODEL
        self.cnn_model = cnn_model
        self.rnd_target = rnd_target
        self.rnd_predictor = rnd_predictor
        self.dqn_model = dqn_model
        self.dqn_target = dqn_target
        
        self.rnd_optimizer = rnd_optimizer
        self.dqn_optimizer = dqn_optimizer        
        self.scaler = GradScaler()

        # MEMORY        
        self.n_memory = MemoryBuffer(buffer_size)
        self.memory = PrioritizedMemoryBuffer(buffer_size)

        # AUX
        self.n_step = 0
        self.l_step = 0
                
        # self.loss = (0.0, 1.0e-10)
        self.rnd_loss = 0
        self.loss = 0
    
    def act(self, state):
        
        action = None
        if np.random.uniform() < self.EPS:
            action = random.choice( np.arange(self.ACTION_SIZE) )

        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.DEVICE)

            self.cnn_model.eval()
            self.dqn_model.eval()

            with torch.no_grad():                
                encoded = self.cnn_model( state )
                action_values = self.dqn_model( encoded )            

            self.dqn_model.train()
            self.cnn_model.train()

            action_values = action_values.cpu().data.numpy()
            action = np.argmax( action_values )
            
        self.EPS *= self.EPS_DECAY
        self.EPS = max(self.EPS_MIN, self.EPS) 

        return action

    def step(self, state, action, reward, next_state, done):
            
        self.n_memory.add( state, action, reward, next_state, done )

        self.n_step = (self.n_step + 1) % self.N_STEPS
        if self.n_step == 0:
            states, actions, rewards, next_states, dones = self.n_memory.exp()

            # discount = self.GAMMA**np.arange( self.N_STEPS )
            # rewards_future = rewards * discount
            # rewards_future = rewards_future[::-1].cumsum(axis=0)[::-1]

            norm_rewards = ( rewards - rewards.mean() ) / rewards.std() + 1e-10

            R = 0
            nR = 0
            for idx in range(self.N_STEPS):
                R += self.GAMMA**idx * rewards[idx]
                nR += self.GAMMA**idx * norm_rewards[idx]
            
            self.memory.add( states[0], actions[0], R, next_states[-1], dones[-1] )
                
        if len( self.memory ) < self.BURNIN:
            return self.rnd_loss, self.loss

        # Learn every UPDATE_EVERY time steps.
        self.l_step = (self.l_step + 1) % self.UPDATE_EVERY
        if self.l_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                rnd_loss, loss = self._learn()
                
                self.rnd_loss = rnd_loss
                self.loss = loss

        return self.rnd_loss, self.loss

    def _learn(self):        
        states, actions, rewards, next_states, dones, importances, sample_indices = self.memory.sample(self.BATCH_SIZE)        


        # TENSORS
        states      = torch.from_numpy( states                 ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions                ).long().to(self.DEVICE)
        rewards     = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE)
        importances = torch.from_numpy( importances            ).float().to(self.DEVICE)


        # RND

        with autocast():                     
            # RND intrinsic reward               
            with torch.no_grad():
                next_encoded = self.cnn_model(next_states)
                rnd_target = self.rnd_target(next_encoded)

            rnd_predictor = self.rnd_predictor( next_encoded )

            Ri = ( torch.sum( ( rnd_target - rnd_predictor ).pow(2), dim=1 ) ) / 2

            # RND Loss
            rnd_loss = Ri.mean()
        
        self.rnd_optimizer.zero_grad()
        self.scaler.scale(rnd_loss).backward() 
        self.scaler.step(self.rnd_optimizer)
        self.scaler.update()
        
        # DQN MUNCHAUSEN

        with autocast(): 
            with torch.no_grad():
                # Get predicted Q values (for next states) from target model
                next_encoded = self.cnn_model( next_states )
                Q_targets_next = self.dqn_target( next_encoded )
                # calculate entropy term with logsum 
                logsum = torch.logsumexp( ( Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) ) / self.ENTROPY_TAU , 1 ).unsqueeze(-1)

                tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.ENTROPY_TAU * logsum
                # target policy
                pi_target = F.softmax( Q_targets_next / self.ENTROPY_TAU, dim=1 )
                Q_target = ( self.GAMMA * ( pi_target * ( Q_targets_next - tau_log_pi_next ) * ( 1 - dones ).unsqueeze(-1) ).sum(1) ).unsqueeze(-1)
                
                # calculate munchausen addon with logsum trick
                encoded = self.cnn_model( states )
                q_k_targets = self.dqn_target( encoded )
                v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
                logsum = torch.logsumexp( ( q_k_targets - v_k_target ) / self.ENTROPY_TAU, 1 ).unsqueeze(-1)
                log_pi = q_k_targets - v_k_target - self.ENTROPY_TAU * logsum
                munchausen_addon = log_pi.gather(1, actions.unsqueeze(-1))
                
                # calc munchausen reward:
                ie_rewards = rewards + Ri.detach() * 0.0001
                munchausen_reward = (ie_rewards.unsqueeze(-1) + self.ALPHA * torch.clamp(munchausen_addon, min=self.LO, max=0))
                
                # Compute Q targets for current states 
                Q_targets = munchausen_reward + Q_target
                
            # Get expected Q values from local model
            encoded = self.cnn_model( states )
            q_k = self.dqn_model( encoded )
            Q_values = q_k.gather(1, actions.unsqueeze(-1))
            
            # Compute loss
            # loss = F.mse_loss(Q_values, Q_targets) # mse_loss
            loss = Q_values - Q_targets
            importance_loss = ( loss ** 2 ) * importances.unsqueeze(-1)
            loss  = importance_loss.mean()

            l2_factor = 1e-8

            l2_reg_cnn = None
            for W in self.cnn_model.parameters():
                if l2_reg_cnn is None:
                    l2_reg_cnn = W.norm(2)
                else:
                    l2_reg_cnn = l2_reg_cnn + W.norm(2)

            l2_reg_cnn = l2_reg_cnn * l2_factor

            loss += l2_reg_cnn

            l2_reg_dqn = None
            for W in self.dqn_model.parameters():
                if l2_reg_dqn is None:
                    l2_reg_dqn = W.norm(2)
                else:
                    l2_reg_dqn = l2_reg_dqn + W.norm(2)

            l2_reg_dqn = l2_reg_dqn * l2_factor

            loss += l2_reg_dqn

        self.dqn_optimizer.zero_grad()
        self.scaler.scale(loss).backward() 
        self.scaler.step(self.dqn_optimizer)
        self.scaler.update()
        

        self.memory.set_priorities( sample_indices, importance_loss.squeeze(1).cpu().data.numpy() )

        self._soft_update_target_model()


        return rnd_loss.cpu().data.numpy().item(), loss.cpu().data.numpy().item()


    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.dqn_target.parameters(), self.dqn_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)        
