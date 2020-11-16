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

from model import CNNModel, DQNModel, ActorModel, CriticModel, RNDTargetModel, RNDPredictorModel

from agent import Agent


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0)

# ORIGINAL
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# BLACK
# env = gym_super_mario_bros.make('SuperMarioBros-v1')
# PIXEL
# env = gym_super_mario_bros.make('SuperMarioBros-v2')
# ONLY FIRST STATE
# env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# RANDOM STAGES
# env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')

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

# 240 W
# 256 H
height_pixel_cut = 25
img_w = int(state_example.shape[0]/2)
img_h = int(state_example.shape[1]/2)


# # actions for very simple movement
# SIMPLE_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
# ]

# HYPERPARAMETERS
T_FRAME_SKIP = 4

LR = 5e-4
EPS = 1
EPS_DECAY = 0.99995
EPS_MIN = 0.1
ENTROPY_TAU = 3e-2
ALPHA = 0.9
LO = -1
GAMMA = 0.95
TAU = 1e-3

N_STEPS = 4
UPDATE_EVERY = 32
BATCH_SIZE = 256

# BURNIN = int(5e3)
BURNIN = BATCH_SIZE * 4
BURNIN = 0

BUFFER_SIZE = int(5e4)


RND_LR = 5e-4
RND_OUTPUT_SIZE = 1024
RND_UPDATE_EVERY = 32


CHECKPOINT_FOLDER = './'

CHECKPOINT_CNN = CHECKPOINT_FOLDER + 'CNN.pth'
CHECKPOINT_RND_TARGET = CHECKPOINT_FOLDER + 'RND_TARGET.pth'
CHECKPOINT_RND_PREDICTOR = CHECKPOINT_FOLDER + 'RND_PREDICTOR.pth'
CHECKPOINT_DQN = CHECKPOINT_FOLDER + 'DQN.pth'

# MODELS
cnn_model = CNNModel().to(DEVICE)

rnd_target = RNDTargetModel( cnn_model.state_size, RND_OUTPUT_SIZE ).to(DEVICE)
rnd_predictor = RNDPredictorModel( cnn_model.state_size, RND_OUTPUT_SIZE ).to(DEVICE)

dqn_model = DQNModel(
        state_size=cnn_model.state_size,
        action_size=action_size,
        fc1_units=256,
        fc2_units=128
    ).to(DEVICE)

dqn_target = DQNModel(
        state_size=cnn_model.state_size,
        action_size=action_size,
        fc1_units=256,
        fc2_units=128
    ).to(DEVICE)

rnd_optimizer = optim.Adam( rnd_predictor.parameters(), lr=RND_LR, weight_decay=1e-4 )
dqn_optimizer = optim.Adam( list(cnn_model.parameters()) + list(dqn_model.parameters()), lr=LR )

cnn_model.load(CHECKPOINT_CNN, DEVICE)
rnd_target.load(CHECKPOINT_RND_TARGET, DEVICE)
rnd_predictor.load(CHECKPOINT_RND_PREDICTOR, DEVICE)
dqn_model.load(CHECKPOINT_DQN, DEVICE)
dqn_target.load(CHECKPOINT_DQN, DEVICE)

# AGENT

agent = Agent(
    DEVICE,
    action_size,
    EPS, EPS_DECAY, EPS_MIN,
    BURNIN, N_STEPS, UPDATE_EVERY, BATCH_SIZE, 
    ENTROPY_TAU, ALPHA, LO, GAMMA, TAU, 
    cnn_model, rnd_target, rnd_predictor, dqn_model, dqn_target, 
    rnd_optimizer, dqn_optimizer,
    BUFFER_SIZE
    )

# TRAIN

def train(n_episodes, height_pixel_cut=15):            

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')

    t_frame = 0    
    for episode in range(n_episodes):                

        s = env.reset()

        state = Image.fromarray(s).resize( ( img_w, img_h ) )
        # state = transforms.functional.to_grayscale(state)
        # tensorToImg( imgToTensor(state) ).save('check.jpg')
        # tensorToImg( imgToTensor(state)[:,height_pixel_cut:,:] ).save('check1.jpg')
        state = imgToTensor(state)[:,height_pixel_cut:,:].cpu().data.numpy()

        # while True:
        steps = 3000
        for step in range(steps):
            
            action, action_values = agent.act( state )
            
            next_state, reward, done, info = env.step(action)
            # reward = np.sign(reward)
            s = next_state

            next_state = Image.fromarray(next_state).resize( ( img_w, img_h ) )
            # next_state = transforms.functional.to_grayscale(next_state)
            # tensorToImg( imgToTensor(state)).save('check2.jpg')
            next_state = imgToTensor(next_state)[:,height_pixel_cut:,:].cpu().data.numpy()
        
            t_frame = (t_frame + 1) % T_FRAME_SKIP
            if t_frame == 0:
                rnd_loss, loss = agent.step( state, action, reward, next_state, done )

                print('\r E: {} STEP: {} POS: {} R: {} RL: {:.10f} L: {:.10f}'.format(
                    episode + 1, step, info['x_pos'], reward, 
                    rnd_loss, loss
                    ), end='')                
            
                render( fig, axs, s, action_values )
            # env.render()            

            state = next_state

            if done or step == 3000:                
                cnn_model.checkpoint( CHECKPOINT_CNN )
                rnd_target.checkpoint( CHECKPOINT_RND_TARGET )
                rnd_predictor.checkpoint( CHECKPOINT_RND_PREDICTOR )
                dqn_model.checkpoint( CHECKPOINT_DQN )

                break

    env.close()


def render(fig, axs, state, action_values):

    x = np.arange( action_size )
    y = action_values.reshape(1, -1)

    axs.clear()
    axs.bar( x, y[0,:], color='red')

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

n_episodes = 1000
train(n_episodes, height_pixel_cut)