import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class Optimizer:

    def __init__(
        self, 
        device,
        memory,
        cnn, actor, critic, optimizer,
        rnd_target, rnd_predictor, rnd_optimizer,
        n_step, gamma, epsilon, entropy_weight
        ):

        self.DEVICE = device
        
        # MEMORY
        self.memory = memory        

        # NEURAL MODEL
        self.cnn = cnn
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

        # RND
        self.rnd_target = rnd_target
        self.rnd_predictor = rnd_predictor        
        self.rnd_optimizer = rnd_optimizer

        # HYPERPARAMETERS
        self.N_STEP = n_step    
        self.GAMMA = gamma
        self.GAMMA_N = gamma ** n_step
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight

        
        self.t_step = 0 
        self.loss = 0
        self.rnd_loss = 0

    def step(self, state, action, log_prob, reward, next_state):
        self.memory.add( state.T, action, log_prob, reward, next_state.T )

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.N_STEP
        if self.t_step == 0:
            self.loss, self.rnd_loss = self._learn()

        return self.loss, self.rnd_loss

        
    def _learn(self):
        states, actions, log_probs, rewards, next_states, n_exp = self.memory.experiences()


        discount = self.GAMMA**np.arange(n_exp)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]    


        states      = torch.from_numpy( states                ).float().to(self.DEVICE)           
        actions     = torch.from_numpy( actions               ).long().to(self.DEVICE)
        log_probs   = torch.from_numpy( log_probs             ).float().to(self.DEVICE)
        rewards     = torch.from_numpy( rewards_future.copy() ).float().to(self.DEVICE)
        next_states = torch.from_numpy( next_states           ).float().to(self.DEVICE)
        

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

        # PPO

        self.optimizer.zero_grad()


        self.critic.eval()
        with torch.no_grad():
            flatten_states = self.cnn( states )
            values = self.critic( flatten_states ).squeeze(1).detach()
        self.critic.train()
                        

        ie_rewards = rewards +  ( Ri.detach() * 0.0001 )
        advantages = (ie_rewards - values).detach()
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        advantages_normalized = torch.tensor(advantages_normalized).float().to(self.DEVICE)
        

        flatten_states = self.cnn( states )


        _, new_log_probs, entropies = self.actor(flatten_states, actions)


        ratio = ( new_log_probs - log_probs ).exp()

        clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

        policy_loss = torch.min( ratio * advantages_normalized, clip * advantages_normalized )
        policy_loss = - torch.mean( policy_loss )

        entropy = torch.mean(entropies)

        
        values = self.critic( flatten_states ).squeeze(1)
        value_loss = F.mse_loss( rewards, values )


        loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)

        loss.backward()
        nn.utils.clip_grad_norm_( self.actor.parameters(), 0.5 )
        nn.utils.clip_grad_norm_( self.critic.parameters(), 0.5 )
        self.optimizer.step()

        return loss.item(), rnd_loss.item()
