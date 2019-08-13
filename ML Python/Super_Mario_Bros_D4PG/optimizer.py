import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class Optimizer:

    def __init__(
        self, 
        device,
        good_memory, bad_memory,
        cnn, 
        actor_model, actor_target, optimizer_actor, 
        critic_model, critic_target, optimizer_critic, 
        rnd_target, rnd_predictor, rnd_optimizer,
        alpha, gamma, TAU, update_every, buffer_size, batch_size
        ):

        self.DEVICE = device
        
        # MEMORY
        self.good_memory = good_memory
        self.bad_memory = bad_memory

        # NEURAL MODEL
        self.cnn = cnn

        self.actor_model = actor_model
        self.actor_target = actor_target
        self.optimizer_actor = optimizer_actor

        self.critic_model = critic_model
        self.critic_target = critic_target
        self.optimizer_critic = optimizer_critic

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
        self.actor_loss = 0
        self.critic_loss = 0
        self.rnd_loss = 0

    def step(self, state, action, reward, next_state, done):
        if reward > 0:
            self.good_memory.add( state.T, action, reward, next_state.T, done )
        else:
            self.bad_memory.add( state.T, action, reward, next_state.T, done )

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory then learn            
            # if self.good_memory.enougth_samples() and self.bad_memory.enougth_samples():
            self.actor_loss, self.critic_loss, self.rnd_loss = self._learn()

        return self.actor_loss, self.critic_loss, self.rnd_loss

        
    def _learn(self):                    
        states              = []
        actions             = []
        rewards             = []
        next_states         = []
        dones               = []

        good_samples = self.good_memory.sample()
        if good_samples:
            t_states, t_actions, t_rewards, t_next_states, t_dones = good_samples

            states.append(t_states)
            actions.append(t_actions)
            rewards.append(t_rewards)
            next_states.append(t_next_states)
            dones.append(t_dones)

        bad_samples = self.bad_memory.sample()
        if bad_samples:
            t_states, t_actions, t_rewards, t_next_states, t_dones = bad_samples

            states.append(t_states)
            actions.append(t_actions)
            rewards.append(t_rewards)
            next_states.append(t_next_states)
            dones.append(t_dones)

        if not good_samples and not bad_samples:
            return 0, 0, 0

        states      = np.vstack( states )
        actions     = np.vstack( actions ).reshape(1, -1)
        rewards     = np.vstack( rewards ).reshape(1, -1)
        next_states = np.vstack( next_states )
        dones       = np.vstack( dones ).reshape(1, -1)


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

            states_flatten = self.cnn(states)
            action_values = self.actor_model(states_flatten)

        rnd_predictor = self.rnd_predictor( next_states_flatten, action_values )

        Ri = ( torch.sum( ( rnd_target - rnd_predictor ).pow(2), dim=1 ) ) / 2

        # RND Loss
        rnd_loss = Ri.mean()
        
        rnd_loss.backward()
        # nn.utils.clip_grad_norm_( self.rnd_predictor.parameters(), 1 )

        self.rnd_optimizer.step()

        # D4PG

        l2_factor = 0.0005

        # CRITIC

        self.optimizer_critic.zero_grad()

        next_states_flatten = self.cnn(next_states)  
        actions_next = self.actor_target(next_states_flatten)

        Q_targets_next = self.critic_target(next_states_flatten, actions_next).squeeze()

        ie_rewards = rewards + Ri.detach() * 0.0001
        ie_rewards = torch.tanh( ie_rewards )
        Q_targets = ie_rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        
        states_flatten = self.cnn(states)
        actions_value = self.actor_model(states_flatten)
        Q_expected = self.critic_model(states_flatten, actions_value).squeeze()
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # L2 Regularization              

        l2_reg_critic = None
        for W in self.critic_model.parameters():
            if l2_reg_critic is None:
                l2_reg_critic = W.norm(2)
            else:
                l2_reg_critic = l2_reg_critic + W.norm(2)

        l2_reg_critic = l2_reg_critic * l2_factor

        critic_loss = critic_loss + l2_reg_critic
        
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_model.parameters(), 1)
        self.optimizer_critic.step()  
        
        # ACTOR

        self.optimizer_actor.zero_grad()

        states_flatten = self.cnn(states)
        actions_pred = self.actor_model(states_flatten)
        actor_loss = - self.critic_model(states_flatten, actions_pred).mean()

        # L2 ACTOR

        l2_reg_actor = None
        for W in self.actor_model.parameters():
            if l2_reg_actor is None:
                l2_reg_actor = W.norm(2)
            else:
                l2_reg_actor = l2_reg_actor + W.norm(2)

        l2_reg_actor = l2_reg_actor * l2_factor

        actor_loss = actor_loss + l2_reg_actor
        
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target model
        self.soft_update_target_model(self.actor_model, self.actor_target)
        self.soft_update_target_model(self.critic_model, self.critic_target)


        return actor_loss.item(), critic_loss.item(), rnd_loss.item()

    def soft_update_target_model(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
    