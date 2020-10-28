import os
import numpy as np
from collections import deque

import random

import cv2 as cv
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from agent_dqn import Agent

from model import DQNModel, ActorModel, CriticModel


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0)

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
height_pixel_cut = 25
img_w = int(state_example.shape[0]/2)
img_h = int(state_example.shape[1]/2)



# HYPERPARAMETERS
T_FRAME_SKIP = 4

EPS = 0
EPS_DECAY = 0.9995
EPS_MIN = 0.1
ENTROPY_TAU = 3e-2
ALPHA = 0.9
LO = -1
GAMMA = 0.99
TAU = 1e-3
LR = 1e-5

N_STEPS = 1
UPDATE_EVERY = 32
BATCH_SIZE = 512

# BURNIN = int(5e3)
BURNIN = BATCH_SIZE * 4


BUFFER_SIZE = int(5e4)

CHECKPOINT_FOLDER = './'

# CHECKPOINT_ACTION = CHECKPOINT_FOLDER + 'ACTION.pth'
CHECKPOINT_DQN = CHECKPOINT_FOLDER + 'DQN.pth'

# MODELS
dqn_model = DQNModel(        
        action_size=action_size,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32
    ).to(DEVICE)

dqn_target = DQNModel(        
        action_size=action_size,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32
    ).to(DEVICE)

optimizer = optim.RMSprop( dqn_model.parameters(), lr=LR )

dqn_model.load(CHECKPOINT_DQN, DEVICE)
dqn_target.load(CHECKPOINT_DQN, DEVICE)

# AGENT

agent = Agent(DEVICE,         
    action_size,
    EPS, EPS_DECAY, EPS_MIN,
    BURNIN, N_STEPS, UPDATE_EVERY, BATCH_SIZE, 
    ENTROPY_TAU, ALPHA, LO, GAMMA, TAU, 
    dqn_model, dqn_target, 
    optimizer,
    BUFFER_SIZE,
    CHECKPOINT_DQN
    )

# TRAIN

def train(n_episodes, height_pixel_cut=15):            

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')

    t_frame = 0    
    t_step = 0
    for episode in range(n_episodes):
        
        total_reward = 0            

        s = env.reset()

        state = Image.fromarray(s).resize( ( img_w, img_h ) )
        # state = transforms.functional.to_grayscale(state)
        # tensorToImg( imgToTensor(state) ).save('check.jpg')
        # tensorToImg( imgToTensor(state)[:,height_pixel_cut:,:] ).save('check1.jpg')
        state = imgToTensor(state)[:,height_pixel_cut:,:].cpu().data.numpy()

        # while True:        
        n_steps = 3000
        for _ in range(n_steps):
            
            dist, action = agent.act( state )

            next_state, reward, done, info = env.step(action)
            s = next_state
            # reward = np.sign(reward)
            # reward = reward if reward <= 0 else 1

            next_state = Image.fromarray(next_state).resize( ( img_w, img_h ) )
            # next_state = transforms.functional.to_grayscale(next_state)
            # tensorToImg( imgToTensor(state)).save('check2.jpg')
            next_state = imgToTensor(next_state)[:,height_pixel_cut:,:].cpu().data.numpy()
        
            t_frame = (t_frame + 1) % T_FRAME_SKIP
            if t_frame == 0:
                loss = agent.step( state, dist, action, reward, next_state, done )
                t_step += 1

                print('\r step:{} E: {} TR: {} R: {} L: {:.5f}'.format( 
                    t_step,
                    episode + 1, total_reward, reward, 
                    loss
                    ), end='')                    

            # env.render()

            total_reward += reward

            state = next_state

            if done:
                render( fig, axs, s, dist )
                agent.checkpoint()
                break                

    env.close()


def render(fig, axs, state, dist):

    axs.clear()
    axs.bar( np.arange( len(RIGHT_ONLY) ), dist[0,:], color='red')

    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8, sep='' )
    img  = img.reshape( fig.canvas.get_width_height()[::-1] + (3,) )

    # img is rgb, convert to opencv's default bgr
    img = cv.resize( cv.cvtColor(img, cv.COLOR_RGB2BGR), (500,500) )
    s = cv.resize( state, (500,500) )
    img = np.hstack( ( s[:,:,::-1], img ) )
   
    # display image with opencv or any operation you like
    cv.imshow("game", img)
    cv.waitKey(1)

n_episodes = 10000
train(n_episodes, height_pixel_cut)