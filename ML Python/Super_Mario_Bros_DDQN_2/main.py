import os
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

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

img_h = int(state_example.shape[0]/3)
img_w = int(state_example.shape[1]/3)


# HYPERPARAMETERS
EPSILON = 0.1

# FRAME_SEQ = 4

LR = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
FRAME_SKIP = 10
UPDATE_EVERY = 32

GAMMA = 0.95
TAU = 1e-3


CHECKPOINT_FOLDER = './'


# AGENT
agent = Agent(DEVICE, 
    action_size, 
    LR,
    BUFFER_SIZE, 
    FRAME_SKIP, UPDATE_EVERY, BATCH_SIZE,
    GAMMA, TAU,
    CHECKPOINT_FOLDER )

# TRAIN
n_episodes = 10000

for episode in range(n_episodes):
    
    total_reward = 0    
    # state = env.reset()

    # # STACK THE INITIAL FRAME TO BEGIN THE EPISODE
    # state = Image.fromarray(state).resize( ( img_h, img_w ) )
    # state = transforms.functional.to_grayscale(state)
    # # tensorToImg( imgToTensor(state)).save('check1.jpg')
    # state = imgToTensor(state).squeeze(0).cpu().data.numpy()    

    # state_frames = deque(maxlen=FRAME_SEQ)
    # for i in range(FRAME_SEQ):
    #     state_frames.append(state)


    state = env.reset()

    state = Image.fromarray(state).resize( ( img_h, img_w ) )
    state = transforms.functional.to_grayscale(state)
    # tensorToImg( imgToTensor(state)).save('check1.jpg')
    state = imgToTensor(state).cpu().data.numpy() 

    while True:        
        # action = env.action_space.sample()                   

        action = agent.act( state, EPSILON )        

        next_state, reward, done, info = env.step(action)

        next_state = Image.fromarray(next_state).resize( ( img_h, img_w ) )
        next_state = transforms.functional.to_grayscale(next_state)
        # tensorToImg( imgToTensor(state)).save('check2.jpg')
        next_state = imgToTensor(next_state).cpu().data.numpy()
        
        loss = agent.step( state, action, reward, next_state, done )

        env.render()

        total_reward += reward

        state = next_state

        if done:
            agent.checkpoint()            

            break                
        
        print('\rE: {} TR: {} R: {} L: {:.15f}'.format( episode + 1, total_reward, reward, loss ), end='')

env.close()
