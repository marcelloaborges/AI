import os
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent import Agent


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
action_size = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))


# STATE UTILS
imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()

state_example = env.reset()

img_h = int(state_example.shape[0]/4)
img_w = int(state_example.shape[1]/4)


# HYPERPARAMETERS
EPSILON = 0.05

FRAME_SEQ = 8
COMPRESSED_FEATURES_SIZE = 128

LR = 1e-4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
UPDATE_EVERY = 8

VAE_SAMPLES = 4

GAMMA = 0.99
TAU = 1e-1


CHECKPOINT_FOLDER = './'


# AGENT
agent = Agent(DEVICE, 
    FRAME_SEQ, COMPRESSED_FEATURES_SIZE, action_size, 
    LR,
    BUFFER_SIZE, 
    UPDATE_EVERY, BATCH_SIZE, 
    VAE_SAMPLES,    
    img_w, img_h,
    GAMMA, TAU,
    CHECKPOINT_FOLDER )

# TRAIN
n_episodes = 1000

for episode in range(n_episodes):
    
    total_reward = 0    
    state = env.reset()

    # STACK THE INITIAL FRAME TO BEGIN THE EPISODE
    state = Image.fromarray(state).resize( ( img_h, img_w ) )
    state = transforms.functional.to_grayscale(state)
    # tensorToImg( imgToTensor(state)).save('check1.jpg')
    state = imgToTensor(state).squeeze(0).cpu().data.numpy()    

    state_frames = deque(maxlen=FRAME_SEQ)
    for i in range(FRAME_SEQ):
        state_frames.append(state)


    state = env.reset()

    while True:        
        # action = env.action_space.sample()        

        action = agent.act( state_frames, EPSILON )        

        next_state, reward, done, info = env.step(action)

        next_state = Image.fromarray(next_state).resize( ( img_h, img_w ) )
        next_state = transforms.functional.to_grayscale(next_state)
        # tensorToImg( imgToTensor(state)).save('check2.jpg')
        next_state = imgToTensor(next_state).squeeze(0).cpu().data.numpy()    

        temp_state_frames = state_frames.copy()
        temp_state_frames.append( next_state )

        vae_loss, dqn_loss, encoder_check = agent.step( state_frames, action, reward, temp_state_frames, done )

        env.render()

        total_reward += reward

        if done:
            agent.checkpoint()

            if encoder_check:
                for i, img in enumerate(encoder_check[0]):
                    tensorToImg(img.cpu()).save('test/{}.jpg'.format(i))        

            break        

        state_frames.append(next_state)
        
        print('\rE: {} TR: {} R: {} VL: {:.5f} QL: {:.5f}'.format( episode + 1, total_reward, reward, vae_loss, dqn_loss ), end='')

env.close()
