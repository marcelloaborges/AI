import os
from os import system
import numpy as np
from PIL import Image

import cv2 as cv
from collections import deque

import torch
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from model import Encoder, Decoder, DDQN, ICMTarget, ICM

from unusual_memory import UnusualMemory
# from memory import Memory
# from prioritized_memory import PrioritizedMemory
from agent import Agent
from optimizer import Optimizer


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

env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_info = env.reset()
action_info = env.action_space.sample()
ACTION_SIZE = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(ACTION_SIZE))


# hyperparameters
BATCH_SIZE = 64

# VAE
COMPRESSED_FEATURES_SIZE = 128
VAE_SAMPLES = 4
VAE_LR = 1e-3
CICLICAL_B_STEPS = 1000

# DQN
ALPHA = 1
GAMMA = 0.9
TAU = 1e-3
UPDATE_EVERY = 16
BUFFER_SIZE = int(1e4)
DDQN_LR = 3e-4
EPSILON = 0.05

# ICM
ICM_LR = 1e-3
ICM_OUTPUT_SIZE = 64

CHECKPOINT_ENCODER = './checkpoint_encoder.pth'
CHECKPOINT_DECODER = './checkpoint_decoder.pth'
CHECKPOINT_DDQN = './checkpoint_ddqn.pth'
CHECKPOINT_ICM_TARGET = './checkpoint_icm_target.pth'
CHECKPOINT_ICM = './checkpoint_icm.pth'


icm_target = ICMTarget( COMPRESSED_FEATURES_SIZE, ICM_OUTPUT_SIZE ).to(DEVICE)
icm = ICM( COMPRESSED_FEATURES_SIZE, ACTION_SIZE, ICM_OUTPUT_SIZE ).to(DEVICE)
icm_optimizer = optim.Adam( icm.parameters(), lr=ICM_LR, weight_decay=1e-4 )

encoder = Encoder(COMPRESSED_FEATURES_SIZE).to(DEVICE)
decoder = Decoder(COMPRESSED_FEATURES_SIZE).to(DEVICE)
vae_optimizer = optim.Adam( list(encoder.parameters()) + list(decoder.parameters()), lr=VAE_LR, weight_decay=1e-4 )

ddqn_model = DDQN( COMPRESSED_FEATURES_SIZE, ACTION_SIZE ).to(DEVICE)
ddqn_target = DDQN( COMPRESSED_FEATURES_SIZE, ACTION_SIZE ).to(DEVICE)
ddqn_optimizer = optim.Adam( list(ddqn_model.parameters()) + list(encoder.parameters()), lr=DDQN_LR, weight_decay=1e-4 )
# ddqn_optimizer = optim.Adam( ddqn_model.parameters(), lr=DDQN_LR, weight_decay=1e-4 )


encoder.load(CHECKPOINT_ENCODER, DEVICE)
decoder.load(CHECKPOINT_DECODER, DEVICE)
ddqn_model.load(CHECKPOINT_DDQN, DEVICE)
ddqn_target.load(CHECKPOINT_DDQN, DEVICE)
icm_target.load(CHECKPOINT_ICM_TARGET, DEVICE)
icm.load(CHECKPOINT_ICM, DEVICE)


memory = UnusualMemory(BUFFER_SIZE, BATCH_SIZE)
# memory = Memory(BUFFER_SIZE, BATCH_SIZE)
# memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)


agent = Agent(DEVICE, encoder, ddqn_model, ACTION_SIZE)
optimizer = Optimizer(
    DEVICE, 
    memory,
    encoder, decoder, ddqn_model, ddqn_target, icm_target, icm,
    vae_optimizer, ddqn_optimizer, icm_optimizer,
    ACTION_SIZE, BATCH_SIZE,
    VAE_SAMPLES, COMPRESSED_FEATURES_SIZE,
    ALPHA, GAMMA, TAU, UPDATE_EVERY)


imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()

img_w = int(256/4)
img_h = int(240/4)

steps = 0
n_episodes = 300

for episode in range(n_episodes):

    total_reward = 0    
    state = env.reset()    
    state = Image.fromarray(state).resize( ( img_w, img_h ) )
    state = imgToTensor(state).cpu().data.numpy()    

    hx = np.zeros( [1, 256] )
    cx = np.zeros( [1, 256] )
    
    while True:        

        # action = env.action_space.sample()        
        action, nhx, ncx = agent.act( state, hx, cx, EPSILON )

        next_state, reward, done, info = env.step(action)
        next_state = Image.fromarray(next_state).resize( ( img_w, img_h ) )
        next_state = imgToTensor(next_state).cpu().data.numpy()    
        
        icm_loss, vae_loss, ddqn_loss, encoder_check = optimizer.step(state, hx, cx, action, reward, next_state, nhx, ncx, done)

        env.render()

        total_reward += reward                

        system('cls')
        print('\rE: {} S: {} TR: {} R: {} ICM: {:.3f} VAE: {:.3f} DDQN: {:.3f}'
            .format( episode + 1, steps, total_reward, reward, icm_loss, vae_loss, ddqn_loss ), end='')

        if steps > 0 and steps % 100 == 0:
            for i, img in enumerate(encoder_check):
                tensorToImg(img.cpu()).save('test/{}.jpg'.format(i))

        if done:
            break

        state = next_state
        hx = nhx
        cx = ncx
        steps += 1

    encoder.checkpoint(CHECKPOINT_ENCODER)
    decoder.checkpoint(CHECKPOINT_DECODER)
    ddqn_model.checkpoint(CHECKPOINT_DDQN)
    icm_target.checkpoint(CHECKPOINT_ICM_TARGET)
    icm.checkpoint(CHECKPOINT_ICM)

env.close()
