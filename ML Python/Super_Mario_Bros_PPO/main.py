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

from model import PPOModel, ActorModel, CriticModel

from agent_ppo import AgentPPO


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

LR = 1e-4
T_STEPS = 768
GAMMA = 0.95
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001

BATCH_SIZE = 32


CHECKPOINT_FOLDER = './knowlegde/'
CHECKPOINT_PPO = CHECKPOINT_FOLDER + 'PPO.pth'
CHECKPOINT_ACTOR = CHECKPOINT_FOLDER + 'ACTOR.pth'
CHECKPOINT_CRITIC = CHECKPOINT_FOLDER + 'CRITIC.pth'


env = env_wrapper.create_train_env( 1, 1, CUSTOM_MOVEMENT, N_STACKED_FRAMES )

orig, state_info = env.reset()
action_info = env.action_space.sample()
action_size = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))


# MODELS
# ppo_model = PPOModel( N_STACKED_FRAMES , action_size ).to(DEVICE)
actor_model = ActorModel( N_STACKED_FRAMES, action_size ).to(DEVICE)
critic_model = CriticModel( N_STACKED_FRAMES ).to(DEVICE)
optimizer = optim.Adam( list(actor_model.parameters()) + list(critic_model.parameters()), lr=LR )

actor_model.load( CHECKPOINT_ACTOR, DEVICE )
critic_model.load( CHECKPOINT_CRITIC, DEVICE )

# AGENT
agent = AgentPPO(
    DEVICE,
    BATCH_SIZE,
    GAMMA, EPSILON, ENTROPY_WEIGHT,
    actor_model, critic_model, optimizer
    )

# TRAIN

def train():            

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')
    ave_pos = deque(maxlen=100)
    flags = deque(maxlen=100)
    episode = 0

    while np.count_nonzero(flags) < 90:

        orig, state = env.reset()

        # while True:        
        for step in range(T_STEPS):
            
            action, probs, log_prob = agent.act( state )
            
            orig, next_state, reward, done, info = env.step(action)
            
            agent.step( state, action, log_prob, reward, next_state, done )            

            render( fig, axs, orig, probs )

            state = next_state

            if done or step == T_STEPS - 1:                
                loss = agent.learn()

                if info['flag_get']:
                    flags.append(1)
                else:
                    flags.append(0)

                ave_pos.append( info['x_pos'] )

                episode += 1

                print('E: {:5} STEP: {:5} POS: {:.0f} R: {:.2f} F: {} L: {:.5f}'.format(
                    episode + 1, step, np.average( ave_pos ), reward, np.count_nonzero(flags), loss))

                break

        actor_model.checkpoint( CHECKPOINT_ACTOR )
        critic_model.checkpoint( CHECKPOINT_CRITIC )

    env.close()

def teste(n_episodes):            

    fig, axs = plt.subplots(1)
    fig.suptitle('Vertically stacked subplots')
    ave_pos = deque(maxlen=100)
    flags = deque(maxlen=100)
    episode = 0

    for episode in range(n_episodes):    

        orig, state = env.reset()
        step = 0

        while True:        
            
            action, probs, _ = agent.act( state )
            
            orig, next_state, reward, done, info = env.step(action)
            
            # agent.step( state, action, log_prob, reward, next_state, done )            

            render( fig, axs, orig, probs )

            state = next_state
            step += 1

            if done:
                if info['flag_get']:
                    flags.append(1)
                else:
                    flags.append(0)

                ave_pos.append( info['x_pos'] )

                episode += 1

                print('E: {:5} STEP: {:5} POS: {:.0f} R: {:.2f} F: {}'.format(
                    episode + 1, step, np.average( ave_pos ), reward, np.count_nonzero(flags) ))

                break

    env.close()


def render(fig, axs, state, probs):

    x = np.arange( action_size )
    # y = probs.reshape(1, -1)
    y = probs

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


# train()

n_episodes = 10
teste(n_episodes)