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
        burnin, q_start_learning, update_every, batch_size, gamma, tau,
        attention_model, action_model, model, target_model,
        attention_optimizer, agent_optimizer,
        buffer_size,
        checkpoint_attention, checkpoint_action, checkpoint_dqn
        ):

        self.DEVICE = device

        # HYPERPARAMETERS           
        self.SEQ_LEN = seq_len
        self.ACTION_SIZE = action_size
        self.EPS = eps
        self.EPS_DECAY = eps_decay
        self.EPS_MIN = eps_min

        self.BURNIN = burnin
        self.Q_START_LEARNING = q_start_learning
        self.UPDATE_EVERY = update_every        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = tau

        # NEURAL MODEL
        self.attention_model = attention_model
        self.action_model = action_model
        self.model = model
        self.target_model = target_model

        self.attention_model.eval()

        self.attention_optimizer = attention_optimizer
        self.agent_optimizer = agent_optimizer
        self.scaler = GradScaler() 

        self.CHECKPOINT_ATTENTION = checkpoint_attention
        self.CHECKPOINT_ACTION = checkpoint_action
        self.CHECKPOINT_DQN = checkpoint_dqn

        # MEMORY
        self.memory = MemoryBuffer(buffer_size)

        # AUX        
        self.t_step = 0        
        self.l_step = 0
        self.q_step = 0
        
        self.attention_loss = (0.0, 1.0e-10)
        self.q_loss = (0.0, 1.0e-10)

        # WANDB
        wandb.init(project="gpt-2", group="exp1_1", job_type="eval")

        wandb.watch(self.attention_model)
        wandb.watch(self.action_model)
        wandb.watch(self.model)
    
    def act(self, state):

        action = None
        dist = None
        if np.random.uniform() < self.EPS:
            action_values = np.random.uniform( -1, 1, self.ACTION_SIZE ).reshape(1, self.ACTION_SIZE)

        else:            
            state = torch.tensor(state).unsqueeze(0).float().to(self.DEVICE)            
            
            self.model.eval()

            with torch.no_grad():            
                encoded = self.attention_model(state)
                action_values = self.model(encoded[:,-1:].squeeze(1))
                        
            self.model.train()
                    
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
            return self.attention_loss[0]/self.attention_loss[1], self.q_loss[0]/self.q_loss[1]

        # Learn every UPDATE_EVERY time steps.
        self.l_step = (self.l_step + 1) % self.UPDATE_EVERY

        if self.l_step == 0:
            if self.memory.enougth_samples(self.BATCH_SIZE):
                attention_loss, q_loss = self._learn()

                self.attention_loss = (self.attention_loss[0] * 0.99 + attention_loss, self.attention_loss[1] * 0.99 + 1.0)
                self.q_loss = (self.q_loss[0] * 0.99 + q_loss, self.q_loss[1] * 0.99 + 1.0)

        return self.attention_loss[0]/self.attention_loss[1], self.q_loss[0]/self.q_loss[1]

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
        
        attention_loss = torch.tensor(0).to(self.DEVICE)
        q_loss = torch.tensor(0).to(self.DEVICE)

        if self.q_step < self.Q_START_LEARNING:
            with autocast(): 

                # GPT2
                encoded = self.attention_model( states )
                predicted_reward = self.action_model( encoded, dists )        

                norm_rewards = ( rewards_future - rewards_future.mean() ) / rewards_future.std() + 1.0e-10

                attention_loss = F.mse_loss( predicted_reward.squeeze(-1), norm_rewards )

                # L2 Regularization
                l2_factor = 1e-8

                l2_reg_attention = None
                for W in self.attention_model.parameters():
                    if l2_reg_attention is None:
                        l2_reg_attention = W.norm(2)
                    else:
                        l2_reg_attention = l2_reg_attention + W.norm(2)

                l2_reg_attention = l2_reg_attention * l2_factor

                attention_loss += l2_reg_attention

                l2_reg_action = None
                for W in self.action_model.parameters():
                    if l2_reg_action is None:
                        l2_reg_action = W.norm(2)
                    else:
                        l2_reg_action = l2_reg_action + W.norm(2)

                l2_reg_action = l2_reg_action * l2_factor

                attention_loss += l2_reg_action                        

            # BACKWARD        
            self.attention_optimizer.zero_grad()
            self.scaler.scale(attention_loss).backward() 
            self.scaler.step(self.attention_optimizer)
            self.scaler.update()

            # WANDB
            pr = predicted_reward.squeeze(-1)[-1][-1]
            r = norm_rewards[-1][-1]

            wandb.log(
                {                            
                    "attention_loss": attention_loss,                    
                    "predicted": pr,
                    "expected": r,
                }
            )

        # DQN            
        if self.q_step >= self.Q_START_LEARNING:        
            with autocast(): 
                with torch.no_grad():
                    encoded_ns = self.attention_model(next_states)[:,-1:].squeeze(1)
                    Q_target_next = self.target_model(encoded_ns)
                    Q_target_next = Q_target_next.max(1)[0]
                    
                    Q_target = rewards[:,-1:].squeeze(1) + self.GAMMA * Q_target_next * (1 - dones[:,-1:].squeeze(1))            
                                            
                with torch.no_grad():
                    encoded_s = self.attention_model(states)[:,-1:].squeeze(1)
                Q_value = self.model(encoded_s)
                Q_value = Q_value.gather(1, actions[:,-1:]).squeeze(1)            

                q_loss = F.mse_loss(Q_value, Q_target)

                l2_factor = 1e-8

                l2_reg_actor = None
                for W in self.model.parameters():
                    if l2_reg_actor is None:
                        l2_reg_actor = W.norm(2)
                    else:
                        l2_reg_actor = l2_reg_actor + W.norm(2)

                l2_reg_actor = l2_reg_actor * l2_factor

                q_loss += l2_reg_actor

            self.agent_optimizer.zero_grad()
            self.scaler.scale(q_loss).backward() 
            self.scaler.step(self.agent_optimizer)
            self.scaler.update()

            # WANDB            
            wandb.log(
                {                                                
                    "q_loss": q_loss                    
                }
            )

        self.q_step += 1        

        return attention_loss.cpu().data.numpy().item(), q_loss.cpu().data.numpy().item()

    def _soft_update_target_model(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.TAU*model_param.data + (1.0-self.TAU)*target_param.data)

    def checkpoint(self):        
        self.attention_model.checkpoint(self.CHECKPOINT_ATTENTION)
        self.action_model.checkpoint(self.CHECKPOINT_ACTION)
        self.model.checkpoint(self.CHECKPOINT_DQN)
