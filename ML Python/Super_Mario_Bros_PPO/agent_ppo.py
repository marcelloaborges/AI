import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer
from prioritized_memory_buffer import PrioritizedMemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class AgentPPO:

    def __init__(
        self, 
        device,                
        batch_size, 
        gamma, epsilon, entropy_weight,
        actor_model, critic_model, optimizer        
        ):

        self.DEVICE = device

        # HYPERPARAMETERS
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = 0.95
        self.EPSILON = epsilon
        self.ENTROPY_WEIGHT = entropy_weight

        # NEURAL MODEL
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer        
        self.scaler = GradScaler()

        # MEMORY
        self.n_memory = MemoryBuffer()
        self.memory = MemoryBuffer()

        # self.loss = (0.0, 1.0e-10)
        self.loss = 0
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.DEVICE)

        self.actor_model.eval()

        with torch.no_grad():                            
            action, probs, log_prob, _ = self.actor_model(state)

        self.actor_model.train()

        action = action.cpu().data.numpy().item()
        probs = probs.cpu().data.numpy()
        log_prob = log_prob.cpu().data.numpy().item()

        return action, probs, log_prob

    def step(self, state, action, log_prob, reward, next_state, done):
        self.n_memory.add( state, action, log_prob, reward, next_state, done )

        # self.n_step = (self.n_step + 1) % self.N_STEPS
        # if self.n_step == 0:
        #     self.loss = self._learn()

        # return self.loss

    def learn(self):  
        
        states, actions, log_probs, rewards, next_states, dones, n_exp = self.n_memory.exp()


        # TENSORS
        states      = torch.from_numpy( states                ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions               ).long().to(self.DEVICE)
        log_probs   = torch.from_numpy( log_probs             ).float().to(self.DEVICE)
        rewards     = torch.from_numpy( rewards               ).float().to(self.DEVICE)
        # rewards     = torch.from_numpy( rewards_future.copy() ).float().to(self.DEVICE)
        next_states = torch.from_numpy( next_states           ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones                 ).float().to(self.DEVICE)        

        # for _ in range(10):
        with autocast():

            self.critic_model.eval()
            with torch.no_grad():
                new_values = self.critic_model( states.squeeze(1) )
                new_next_values = self.critic_model( next_states.squeeze(1) )
            self.critic_model.train()            
                        
            gae = 0
            R = []
            for idx in reversed( range(n_exp) ):
                gae = gae * self.GAMMA * self.TAU
                gae = gae + rewards[idx] + self.GAMMA * new_next_values.squeeze(1)[idx] * (1 - dones[idx]) - new_values.squeeze(1)[idx]                    
                R.append(gae + new_values[idx])

            R = torch.tensor(R).float().to(self.DEVICE)            
  
            advantages = R.flip(0) - new_values.squeeze(1)
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)
            

            batches = BatchSampler( SubsetRandomSampler( range(0, n_exp) ), self.BATCH_SIZE, drop_last=False)

            for batch_indices in batches:
                batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)
                
                _, _, new_log_probs, entropies = self.actor_model( states[batch_indices].squeeze(1), actions[batch_indices] )


                ratio = ( new_log_probs - log_probs[batch_indices] ).exp()

                clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

                policy_loss = torch.min( ratio * advantages_normalized[batch_indices], clip * advantages_normalized[batch_indices] )
                policy_loss = - torch.mean( policy_loss )

                entropy = torch.mean(entropies)

                
                new_values = self.critic_model( states[batch_indices].squeeze(1) )
                value_loss = F.smooth_l1_loss( R[batch_indices], new_values.squeeze(1) )


                loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)                  


                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()


        return loss.cpu().data.numpy().item()

    def learn2(self):  
        
        states, actions, log_probs, rewards, next_states, dones, n_exp = self.n_memory.exp()

        discount = self.GAMMA**np.arange(n_exp)
        rewards = rewards * discount
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # TENSORS
        states      = torch.from_numpy( states                ).float().to(self.DEVICE)
        actions     = torch.from_numpy( actions               ).long().to(self.DEVICE)
        log_probs   = torch.from_numpy( log_probs             ).float().to(self.DEVICE)
        # rewards     = torch.from_numpy( rewards               ).float().to(self.DEVICE)
        rewards     = torch.from_numpy( rewards_future.copy() ).float().to(self.DEVICE)
        next_states = torch.from_numpy( next_states           ).float().to(self.DEVICE)
        dones       = torch.from_numpy( dones                 ).float().to(self.DEVICE)        

        # for _ in range(10):
        with autocast():

            self.critic_model.eval()
            with torch.no_grad():
                new_values = self.critic_model( states.squeeze(1) )
            self.critic_model.train()            
            
            advantages = rewards - new_values.squeeze(1)
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1.0e-10)            

            batches = BatchSampler( SubsetRandomSampler( range(0, n_exp) ), self.BATCH_SIZE, drop_last=False)

            for batch_indices in batches:
                batch_indices = torch.tensor(batch_indices).long().to(self.DEVICE)
                
                _, _, new_log_probs, entropies = self.actor_model( states[batch_indices].squeeze(1), actions[batch_indices] )


                ratio = ( new_log_probs - log_probs[batch_indices] ).exp()

                clip = torch.clamp( ratio, 1 - self.EPSILON, 1 + self.EPSILON )

                policy_loss = torch.min( ratio * advantages_normalized[batch_indices], clip * advantages_normalized[batch_indices] )
                policy_loss = - torch.mean( policy_loss )

                entropy = torch.mean(entropies)

                
                new_values = self.critic_model( states[batch_indices].squeeze(1) )
                value_loss = F.smooth_l1_loss( rewards[batch_indices], new_values.squeeze(1) )


                loss = policy_loss + (0.5 * value_loss) - (entropy * self.ENTROPY_WEIGHT)                  


                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()


        self.loss = loss.cpu().data.numpy().item()