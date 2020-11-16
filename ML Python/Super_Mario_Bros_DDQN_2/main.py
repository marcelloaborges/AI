import os
import numpy as np
from collections import deque

import random

import cv2
import matplotlib.pyplot as plt

import env_wrapper

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from model import DDQNModel

from agent_ddqn import AgentDDQN


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
# env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')

from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# actions for the simple run right environment
CUSTOM_MOVEMENT = [
    ['NOOP'], # NADA        
    ['right', 'B'], # CORRER
    ['right', 'A', 'B'], # CORRER + PULAR
]

# HYPERPARAMETERS
N_STACKED_FRAMES = 4
T_STEPS = 512

LR = 1e-4
EPS = 1
EPS_DECAY = 0.9995
EPS_MIN = 0.1
ENTROPY_TAU = 3e-2
ALPHA = 0.9
LO = -1
GAMMA = 0.95
TAU = 1e-3

UPDATE_EVERY = 32
BATCH_SIZE = 256

# BURNIN = int(5e3)
BURNIN = BATCH_SIZE * 4
BURNIN = 0

BUFFER_SIZE = int(5e4)


CHECKPOINT_FOLDER = './knowlegde/'
CHECKPOINT_DDQN = CHECKPOINT_FOLDER + 'DDQN.pth'


env = env_wrapper.create_train_env( 1, 1, CUSTOM_MOVEMENT, N_STACKED_FRAMES )

orig, state_info = env.reset()
action_info = env.action_space.sample()
action_size = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))


# MODELS
dqn_model = DDQNModel( N_STACKED_FRAMES, action_size ).to(DEVICE)
dqn_target = DDQNModel( N_STACKED_FRAMES, action_size ).to(DEVICE)
optimizer = optim.Adam( dqn_model.parameters() , lr=LR )

dqn_model.load( CHECKPOINT_DDQN, DEVICE )
dqn_target.load( CHECKPOINT_DDQN, DEVICE )

# AGENT
agent = AgentDDQN(
    DEVICE,
    action_size,
    EPS, EPS_DECAY, EPS_MIN,
    BURNIN, UPDATE_EVERY, BATCH_SIZE,
    ENTROPY_TAU, ALPHA, LO, GAMMA, TAU,
    dqn_model, dqn_target, 
    optimizer,
    BUFFER_SIZE
    )

# TRAIN

def train(n_episodes):            

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')
    ave_pos = deque(maxlen=100)
    flags = deque(maxlen=100)

    for episode in range(n_episodes):

        orig, state = env.reset()

        # while True:        
        for step in range(T_STEPS):
            
            action, action_values = agent.act( state )
            
            orig, next_state, reward, done, info = env.step(action)
            
            loss = agent.step( state, action, reward, next_state, done )

            render( fig, axs, orig, action_values )

            state = next_state

            if done or step == T_STEPS - 1:
                if info['flag_get']:
                    flags.append(1)
                else:
                    flags.append(0)

                ave_pos.append( info['x_pos'] )

                print('E: {:5} STEP: {:5} POS: {:.0f} R: {:5.3f} F: {} L: {:2.5f}'.format(
                    episode + 1, step, np.average( ave_pos ), reward, np.count_nonzero(flags), loss))

                break

        dqn_model.checkpoint( CHECKPOINT_DDQN )

    env.close()


def render(fig, axs, state, values):

    x = np.arange( action_size )
    # y = values.reshape(1, -1)
    y = values

    axs.clear()
    axs.bar( x, y[0,:], color='red')

    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8, sep='' )
    img  = img.reshape( fig.canvas.get_width_height()[::-1] + (3,) )

    # img is rgb, convert to opencv's default bgr
    img = cv2.resize( cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (300,300) )
    s = cv2.resize( state, (300,300) )
    img = np.hstack( ( s[:,:,::-1], img ) )    
   
    # display image with opencv or any operation you like
    cv2.imshow("game", img)
    cv2.waitKey(1)


n_episodes = 5000
train(n_episodes)