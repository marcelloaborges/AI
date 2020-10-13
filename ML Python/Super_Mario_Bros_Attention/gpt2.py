import numpy as np

import random

import pickle, os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from memory_buffer import MemoryBuffer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import wandb

class GPT2:

    def __init__(self, 
        device,                
        batch_folder,
        attention_model,
        action_model,
        optimizer,
        checkpoint_attention,
        checkpoint_action
        ):

        self.DEVICE = device        
        
        self.CHECKPOINT_ATTENTION = checkpoint_attention
        self.CHECKPOINT_ACTION = checkpoint_action
        self.BATCH_FOLDER = batch_folder

        self.attention_model = attention_model
        self.action_model = action_model        

        self.attention_model.train()
        self.action_model.train()
        
        self.optimizer = optimizer        
        self.scaler = GradScaler() 

        wandb.init(project="gpt-2", group="exp1_1", job_type="eval")

        wandb.watch(self.attention_model)
        wandb.watch(self.action_model)

    def train(self, epoches):
        n_epoches = epoches
        avg_loss = (0.0, 0.0)
        
        for epoch in range(n_epoches):                        
            
            training_batch = None
            for _, _, files in os.walk( self.BATCH_FOLDER ):

                random.shuffle(files)

                for batch in files:                    
                    with open('{}/{}'.format(self.BATCH_FOLDER, batch), 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        training_batch = unpickler.load()

                    # TRAIN WITH THE ACTUAL BATCH
                    random.shuffle(training_batch)

                    states      = []
                    actions     = []
                    rewards     = []
                    next_states = []
                    dones       = []

                    for exp in training_batch:
                        states.append     ( exp['states']      )           
                        actions.append    ( exp['actions']     )
                        rewards.append    ( exp['rewards']     )
                        next_states.append( exp['next_states'] )
                        dones.append      ( exp['dones']       )

                    states      = np.array(states)
                    actions     = np.array(actions)
                    rewards     = np.array(rewards)
                    next_states = np.array(next_states)
                    dones       = np.array(dones)

                    discount = 0.9**np.arange( 24 )
                    rewards = rewards * discount
                    rewards_future = rewards[::-1].cumsum(axis=1)[::-1]

                    states      = torch.from_numpy( states                 ).float().to(self.DEVICE)
                    actions     = torch.from_numpy( actions                ).float().to(self.DEVICE).squeeze(2)
                    rewards     = torch.from_numpy( rewards_future.copy()  ).float().to(self.DEVICE)
                    next_states = torch.from_numpy( next_states            ).float().to(self.DEVICE)
                    dones       = torch.from_numpy( dones.astype(np.uint8) ).float().to(self.DEVICE)

                    # TRANSFORMER
                    with autocast(): 
                        encoded = self.attention_model( states )
                        predicted_reward = self.action_model( encoded, actions )        

                        rewards = ( rewards - rewards.mean() ) / rewards.std() + 1.0e-10                        

                        loss = F.mse_loss( predicted_reward.squeeze(-1), rewards )

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

                        l2_reg_action = None
                        for W in self.action_model.parameters():
                            if l2_reg_action is None:
                                l2_reg_action = W.norm(2)
                            else:
                                l2_reg_action = l2_reg_action + W.norm(2)

                        l2_reg_action = l2_reg_action * l2_factor

                        loss += l2_reg_action

                        # BACKWARD
                        # self.optimizer.zero_grad()
                        # loss.backward() 
                        # self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward() 
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
                    pr = predicted_reward.squeeze(-1)[-1][-1]
                    r = rewards[-1][-1]
                                                        
                    print('\rE: {} L: {:.10f} PR: {:.5f} R: {:.5f}'.format( epoch + 1, avg_loss[0]/avg_loss[1], pr, r ), end='')

                    wandb.log(
                        {                            
                            "loss": avg_loss[0]/avg_loss[1],
                            "predicted": pr,
                            "expected": r,
                        }
                    )

                self.attention_model.checkpoint(self.CHECKPOINT_ATTENTION)
                self.action_model.checkpoint(self.CHECKPOINT_ACTION)
