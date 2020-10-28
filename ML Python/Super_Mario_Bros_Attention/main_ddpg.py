import os
import numpy as np
from collections import deque

import random

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from agent_ddpg import Agent

from model import AttentionEncoderModel, AttentionActionModel, DQNModel, ActorModel, CriticModel


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ORIGINAL
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# BLACK
# env = gym_super_mario_bros.make('SuperMarioBros-v1')
# PIXEL
# env = gym_super_mario_bros.make('SuperMarioBros-v2')
# ONLY FIRST STATE
# env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# RANDOM STAGES
env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')

env = JoypadSpace(env, RIGHT_ONLY)

state_info = env.reset()
action_info = env.action_space.sample()
action_size = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))


# STATE UTILS
imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()

state_example = env.reset()

# 240 W
# 256 H
height_pixel_cut = 15
img_w = int(state_example.shape[0]/3)
img_h = int(state_example.shape[1]/3)



# HYPERPARAMETERS
T_FRAME_SKIP = 4

EPS = 0.5
EPS_DECAY = 0.9995
EPS_MIN = 0.1
GAMMA = 0.9
TAU = 1e-3
LR = 25e-5

UPDATE_EVERY = 8
BATCH_SIZE = 32

# BURNIN = int(5e3)
BURNIN = BATCH_SIZE * 64
Q_START_LEARNING = 500


BURNIN = 0
# Q_START_LEARNING = 1



SEQ_LEN = 64
ATTENTION_HEADS = 2
N_ATTENTION_BLOCKS = 4
FC1_UNITS = 4096
FC2_UNITS = 3072
FC3_UNITS = 2048
FC4_UNITS = 1024
IMG_EMBEDDING = 512
ATTENTION_SIZE = 256
ATTENTION_EMBEDDING = 256

BUFFER_SIZE = int(5e4)

CHECKPOINT_FOLDER = './'

CHECKPOINT_ATTENTION = CHECKPOINT_FOLDER + 'ATTENTION.pth'
CHECKPOINT_ATTENTION_ACTION = CHECKPOINT_FOLDER + 'ATTENTION_ACTION.pth'
CHECKPOINT_ACTOR = CHECKPOINT_FOLDER + 'ACTOR.pth'
CHECKPOINT_CRITIC = CHECKPOINT_FOLDER + 'CRITIC.pth'

BATCH_FOLDER = CHECKPOINT_FOLDER + 'batch/'

# MODELS
attention_model = AttentionEncoderModel(
        seq_len=SEQ_LEN,       
        attention_heads=ATTENTION_HEADS, 
        n_attention_blocks=N_ATTENTION_BLOCKS,
        img_h=img_h - height_pixel_cut,
        img_w=img_w,
        fc1_units=FC1_UNITS,
        fc2_units=FC2_UNITS,
        fc3_units=FC3_UNITS,
        fc4_units=FC4_UNITS,
        img_embedding=IMG_EMBEDDING,
        attention_size=ATTENTION_SIZE,
        attention_embedding=ATTENTION_EMBEDDING,        
        device=DEVICE
    ).to(DEVICE)


attention_action_model = AttentionActionModel(
        encoding_size=ATTENTION_EMBEDDING,
        action_size=action_size,
        fc1_units=64,
        fc2_units=32,
        device=DEVICE
    ).to(DEVICE)

actor_model = ActorModel(
        encoding_size=ATTENTION_EMBEDDING,
        action_size=action_size,
        fc1_units=128,
        fc2_units=64,
        device=DEVICE        
    ).to(DEVICE)

actor_target = ActorModel(
        encoding_size=ATTENTION_EMBEDDING,
        action_size=action_size,
        fc1_units=128,
        fc2_units=64,
        device=DEVICE
    ).to(DEVICE)

critic_model = CriticModel(
        encoding_size=ATTENTION_EMBEDDING,
        action_size=action_size,
        fc1_units=128,
        fc2_units=64,
        device=DEVICE
    ).to(DEVICE)

critic_target = CriticModel(
        encoding_size=ATTENTION_EMBEDDING,
        action_size=action_size,
        fc1_units=128,
        fc2_units=64,
        device=DEVICE
    ).to(DEVICE)

# actor_optimizer = optim.Adam( actor_model.parameters(), lr=LR )
optimizer = optim.RMSprop( 
    list(attention_model.parameters()) + 
    list(attention_action_model.parameters()) +     
    list(actor_model.parameters()) + 
    list(critic_model.parameters()), 
    lr=LR )

attention_model.load(CHECKPOINT_ATTENTION, DEVICE)
attention_model.load(CHECKPOINT_ATTENTION_ACTION, DEVICE)
actor_model.load(CHECKPOINT_ACTOR, DEVICE)
actor_target.load(CHECKPOINT_ACTOR, DEVICE)
critic_model.load(CHECKPOINT_CRITIC, DEVICE)
critic_target.load(CHECKPOINT_CRITIC, DEVICE)

# AGENT

agent = Agent(DEVICE,     
    SEQ_LEN,
    action_size,
    EPS, EPS_DECAY, EPS_MIN,
    BURNIN, UPDATE_EVERY, BATCH_SIZE, GAMMA, TAU, 
    attention_model, attention_action_model, 
    actor_model, actor_target, 
    critic_model, critic_target,
    optimizer,
    BUFFER_SIZE,
    CHECKPOINT_ATTENTION, CHECKPOINT_ACTOR, CHECKPOINT_CRITIC
    )

# TRAIN

def train(n_episodes, height_pixel_cut=15):
        
    t_frame = 0    
    t_step = 0
    for episode in range(n_episodes):
        
        total_reward = 0            

        state = env.reset()

        state = Image.fromarray(state).resize( ( img_w, img_h ) )
        state = transforms.functional.to_grayscale(state)
        # self.tensorToImg( self.imgToTensor(state) ).save('check.jpg')
        # self.tensorToImg( self.imgToTensor(state)[:,height_pixel_cut:,:] ).save('check1.jpg')
        state = imgToTensor(state)[:,height_pixel_cut:,:].cpu().data.numpy() 

        seq_state = deque( maxlen=SEQ_LEN )
        seq_dist = deque( maxlen=SEQ_LEN )
        seq_action = deque( maxlen=SEQ_LEN )
        seq_reward = deque( maxlen=SEQ_LEN )
        seq_done = deque( maxlen=SEQ_LEN )
        for _ in range(SEQ_LEN):
            seq_state.append( np.zeros( state.shape ) )
            seq_dist.append( np.zeros( (1, action_size) ) )
            seq_action.append( 0 )
            seq_reward.append( 0 )
            seq_done.append( False )        

        seq_state.append( state )

        # while True:
        n_steps = 1000
        for _ in range(n_steps):
            
            dist, action = agent.act( seq_state )

            next_state, reward, done, _ = env.step(action)
            # reward = np.sign(reward)
            reward = reward if reward <= 0 else 1

            next_state = Image.fromarray(next_state).resize( ( img_w, img_h ) )
            next_state = transforms.functional.to_grayscale(next_state)
            # tensorToImg( imgToTensor(state)).save('check2.jpg')
            next_state = imgToTensor(next_state)[:,height_pixel_cut:,:].cpu().data.numpy()                

            seq_next_state = seq_state.copy()
            seq_next_state.append( next_state )

            seq_dist.append( dist )
            seq_action.append( action )
            seq_reward.append( reward )
            seq_done.append( done )

            t_frame = (t_frame + 1) % T_FRAME_SKIP
            if t_frame == 0:
                attention_loss, actor_loss, critic_loss, loss = \
                    agent.step( seq_state, seq_dist, seq_action, seq_reward, seq_next_state, seq_done )
                t_step += 1

                print('\r step:{} E: {} TR: {} R: {} ATL: {:.5f} AL: {:.5f} CL: {:.5f} L: {:.5f}'.format( 
                    t_step,
                    episode + 1, total_reward, reward, 
                    attention_loss, actor_loss, critic_loss, loss
                    ), end='')

            env.render()

            total_reward += reward

            seq_state.append( next_state )

            if done:
                agent.checkpoint()
                break                

    env.close()

n_episodes = 1000
train(n_episodes)