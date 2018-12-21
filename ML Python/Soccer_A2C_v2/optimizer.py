import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory

class Optimizer:

    def __init__(self, 
        device,        
        actor_model,        
        critic_model,        
        lr,
        n_step,
        gamma,        
        batch_size,                    
        epsilon,
        beta):

        self.DEVICE = device        

        # Neural model
        self.actor_model = actor_model        
        self.critic_model = critic_model
        self.optimizer = optim.Adam( list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr )

        # Memory
        self.nstep_memory = Memory()
        self.optim_memory = Memory()

        # Hyperparameters
        self.N_STEP = n_step
        self.GAMMA = gamma
        self.BATCH_SIZE = batch_size        
        self.epsilon = epsilon
        self.beta = beta

        self.t_step = 0
        self.loss = 0
        
    def step(self, 
        state, 
        teammate_state,
        action, 
        action_prob, 
        reward):        
        
        # Save experience / reward
        self.nstep_memory.add( state, teammate_state, action, action_prob, reward )

        self.t_step = (self.t_step + 1) % self.N_STEP  
        if self.t_step == 0:

            states, teammate_states, actions, actions_probs, rewards = self.nstep_memory.experiences()

            # Calc the discounted rewards
            discounts = self.GAMMA ** np.arange( self.N_STEP )

            rewards = np.asarray( rewards ) * discounts[:,np.newaxis]
            future_rewards = rewards[::-1].cumsum(axis=0)[::-1]  

            # copy the local experiences to the shared memory with the future reward adjustment
            for i in range( self.N_STEP ):
                self.optim_memory.add( states[i], teammate_states[i], actions[i], actions_probs[i], future_rewards[i] )
        
            experiences = self.optim_memory.experiences()
            self._learn(experiences)

        # if self.optim_memory.enough_experiences(self.BATCH_SIZE):
        #     experiences = self.optim_memory.experiences()            
        #     self._learn(experiences)

        return self.loss
            
    def _learn(self, experiences):
        # Get the experiences from the actor and clear the memory        
        # The rewards have the discount already applied
        states, teammate_states, actions, actions_probs, rewards = experiences

        # To tensor
        states = torch.from_numpy(states).float().to(self.DEVICE)
        teammate_states = torch.from_numpy(teammate_states).float().to(self.DEVICE)
        actions = torch.from_numpy(actions).long().to(self.DEVICE)
        actions_probs = torch.from_numpy(actions_probs).float().to(self.DEVICE)
        rewards = torch.from_numpy(rewards).float().to(self.DEVICE)
        
        # policy loss

        # advantage        
        _, new_probs, entropy = self.actor_model(states, actions)        

        self.critic_model.eval()
        with torch.no_grad():
            values = self.critic_model(states, teammate_states)
        self.critic_model.train()        
                
        advantages = rewards - values
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
        
        # ratio = (new_probs - actions_probs).exp()
        ratio = new_probs / actions_probs

        ratio_clipped = torch.clamp( ratio, 1 - self.epsilon, 1 + self.epsilon )
        objective = torch.min( ratio * advantages_normalized, ratio_clipped * advantages_normalized )        

        policy_loss = - torch.mean( objective )

        entropy = torch.mean( entropy )       
        
        # critic loss          
        values = self.critic_model( states, teammate_states )
        value_loss = F.mse_loss( rewards, values )        

        # total loss        
        loss = policy_loss + (0.5 * value_loss) - (entropy * self.beta)

        # optimize
        self.optimizer.zero_grad()        
        loss.backward()
        self.optimizer.step()

        # policy_loss_value = policy_loss.cpu().detach().numpy().squeeze().item()
        # value_loss_value = value_loss.cpu().detach().numpy().squeeze().item()

        # print('\rPolicy loss: \t{:.8f} \tValue loss: \t{:.8f}'.format(policy_loss_value, value_loss_value), end="")  
        # if np.isnan(policy_loss_value) or np.isnan(value_loss_value):
            # print("policy_loss_value: {:.6f}, value_loss_value: {:.6f}".format(policy_loss_value, value_loss_value))        
        
        self.epsilon *= 1
        self.beta *= 0.995

        self.loss = loss