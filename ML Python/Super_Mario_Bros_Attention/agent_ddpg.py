import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import wandb

class Agent:

    def __init__(
        self, 
        device,
        seq_len,
        action_size,        
        eps, eps_decay, eps_min,
        burnin, update_every, batch_size, gamma, tau,
        attention_model, attention_action_model, 
        actor_model, actor_target, 
        critic_model, critic_target,
        optimizer, 
        buffer_size,
        checkpoint_attention, checkpoint_actor, checkpoint_critic
        ):

        self.DEVICE = device

        # HYPERPARAMETERS           
        self.SEQ_LEN = seq_len
        self.ACTION_SIZE = action_size
        self.EPS = eps
        self.EPS_DECAY = eps_decay
        self.EPS_MIN = eps_min

        self.BURNIN = burnin        
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau

        # NEURAL MODEL
        self.attention_model = attention_model
        self.attention_action_model = attention_action_model
        self.actor_model = actor_model
        self.actor_target = actor_target
        self.critic_model = critic_model
        self.critic_target = critic_target        

        self.optimizer = optimizer
        self.scaler = GradScaler()

        self.CHECKPOINT_ATTENTION = checkpoint_attention
        self.CHECKPOINT_ACTOR = checkpoint_actor
        self.CHECKPOINT_CRITIC = checkpoint_critic

        # MEMORY
        self.memory = MemoryBuffer(buffer_size)

        # AUX        
        self.t_step = 0        
        self.l_step = 0
        self.q_step = 0

        self.attention_loss = (0.0, 1.0e-10)        
        self.actor_loss = (0.0, 1.0e-10)
        self.critic_loss = (0.0, 1.0e-10)
        self.loss = (0.0, 1.0e-10)

        # WANDB
        wandb.init(project="gpt-2", group="exp1_1", job_type="eval")

        wandb.watch(self.attention_model)
        wandb.watch(self.attention_action_model)
        wandb.watch(self.actor_model)
        wandb.watch(self.critic_model)
    
    def act(self, state):

        action = None
        dist = None
        if np.random.uniform() < self.EPS:
            action_values = np.random.uniform( -3, 3, self.ACTION_SIZE ).reshape(1, self.ACTION_SIZE)

        else:            
            state = torch.tensor(state).unsqueeze(0).float().to(self.DEVICE)
            
            self.attention_model.eval()
            self.actor_model.eval()

            with torch.no_grad():            
                encoded = self.attention_model(state)
                action_values = self.actor_model(encoded[:,-1:].squeeze(1))

            self.attention_model.train()            
            self.actor_model.train()
                    
            action_values = action_values.cpu().data.numpy()            
        
        dist = action_values
        action = np.argmax( action_values )

        self.EPS *= self.EPS_DECAY
        self.EPS = max(self.EPS_MIN, self.EPS)        
            
        return dist, action

    def step(self, state, dist, action, reward, next_state, done):
        self.memory.add( state, dist, action, reward, next_state, done )        
        
        # Increment step
        self.t_step += 1

        if self.t_step < self.BURNIN:
            return self.attention_loss[0]/self.attention_loss[1], \
                self.actor_loss[0]/self.actor_loss[1], \
                self.critic_loss[0]/self.critic_loss[1], \
                self.loss[0]/self.loss[1]

        # Learn every UPDATE_EVERY time steps.
        self.l_step = (self.l_step + 1) % self.UPDATE_EVERY

        if self.l_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                attention_loss, actor_loss, critic_loss, loss = self._learn()
                
                self.attention_loss = (self.attention_loss[0] * 0.99 + attention_loss, self.attention_loss[1] * 0.99 + 1.0)
                self.actor_loss = (self.actor_loss[0] * 0.99 + actor_loss, self.actor_loss[1] * 0.99 + 1.0)
                self.critic_loss = (self.critic_loss[0] * 0.99 + critic_loss, self.critic_loss[1] * 0.99 + 1.0)
                self.loss = (self.loss[0] * 0.99 + loss, self.loss[1] * 0.99 + 1.0)

        return self.attention_loss[0]/self.attention_loss[1], \
            self.actor_loss[0]/self.actor_loss[1], \
            self.critic_loss[0]/self.critic_loss[1], \
            self.loss[0]/self.loss[1]

    def _learn(self):
        states, dists, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

        # TEMPORAL CORRELATION BETWEEN REWARDS

        discount = 0.9**np.arange( self.SEQ_LEN )
        rewards_future = rewards * discount
        rewards_future = rewards_future[::-1].cumsum(axis=1)[::-1]

        # TENSORS

        states         = torch.from_numpy( states                 ).float().to(self.DEVICE)
        dists          = torch.from_numpy( dists                  ).float().to(self.DEVICE).squeeze(2)
        actions        = torch.from_numpy( actions                ).long().to(self.DEVICE)
        rewards        = torch.from_numpy( rewards                ).float().to(self.DEVICE)
        rewards_future = torch.from_numpy( rewards_future.copy()  ).float().to(self.DEVICE)
        next_states    = torch.from_numpy( next_states            ).float().to(self.DEVICE)
        dones          = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE)
            
        with autocast(): 
                        
            # GPT2
            encoded = self.attention_model( states )
            predicted_reward = self.attention_action_model( encoded, dists )        

            norm_rewards = ( rewards_future - rewards_future.mean() ) / rewards_future.std() + 1.0e-10

            attention_loss = F.mse_loss( predicted_reward.squeeze(-1), norm_rewards )


            # DDPG
            encoded = self.attention_model( states )[:,-1:].squeeze(1)
            encoded_ns = self.attention_model( next_states )[:,-1:].squeeze(1)

            # CRITIC                
            dist_next = self.actor_target( encoded_ns )
            Q_target_next = self.critic_target( encoded_ns, dist_next ).squeeze(1)
            Q_target = rewards[:,-1:].squeeze(1) + self.GAMMA * Q_target_next * (1 - dones[:,-1:].squeeze(1))
            
            Q_expected = self.critic_model( encoded, dists[:,-1:].squeeze(1) ).squeeze(1)

            critic_loss = F.mse_loss(Q_expected, Q_target)
            

            # ACTOR                
            dist_pred = self.actor_model( encoded )
            actor_loss = - self.critic_model( encoded, dist_pred ).mean()


            # LOSS
            loss = attention_loss + actor_loss + 0.5 * critic_loss


            # L2 Regularization
            l2_factor = 1e-8

            l2_reg_attention = None
            for W in self.attention_model.parameters():
                if l2_reg_attention is None:
                    l2_reg_attention = W.norm(2)
                else:
                    l2_reg_attention = l2_reg_attention + W.norm(2)

            l2_reg_attention = l2_reg_attention * l2_factor

            loss += l2_reg_attention

            l2_reg_actor = None
            for W in self.actor_model.parameters():
                if l2_reg_actor is None:
                    l2_reg_actor = W.norm(2)
                else:
                    l2_reg_actor = l2_reg_actor + W.norm(2)

            l2_reg_actor = l2_reg_actor * l2_factor

            loss += l2_reg_actor

            l2_reg_critic = None
            for W in self.critic_model.parameters():
                if l2_reg_critic is None:
                    l2_reg_critic = W.norm(2)
                else:
                    l2_reg_critic = l2_reg_critic + W.norm(2)

            l2_reg_critic = l2_reg_critic * l2_factor
            
            loss += l2_reg_critic

        # BACKWARD        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward() 
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._soft_update_target_model()

        # WANDB
        wandb.log(
            {                            
                "attention_loss": attention_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "loss": loss
            }
        )

        return attention_loss.cpu().data.numpy().item(), \
            actor_loss.cpu().data.numpy().item(), \
            critic_loss.cpu().data.numpy().item(), \
            loss.cpu().data.numpy().item()

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.actor_target.parameters(), self.actor_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

        for target_param, model_param in zip(self.critic_target.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.attention_model.checkpoint(self.CHECKPOINT_ATTENTION)
        self.actor_model.checkpoint(self.CHECKPOINT_ACTOR)
        self.critic_model.checkpoint(self.CHECKPOINT_CRITIC)
