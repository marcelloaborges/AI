import os
import numpy as np

import time
import random

import torch
import torch.optim as optim

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

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


# hyperparameters
ALPHA = 1
GAMMA = 0.9
TAU = 3e-2
UPDATE_EVERY = 16
BUFFER_SIZE = int(5e3)
BATCH_SIZE = 32
LR = 5e-4
EPSILON = 0.05

RND_LR = 5e-5
RND_OUTPUT_SIZE = 128
RND_UPDATE_EVERY = 32

CHECKPOINT_CNN = './checkpoint_cnn.pth'
CHECKPOINT_MODEL = './checkpoint_model.pth'
CHECKPOINT_RND_TARGET = './checkpoint_rnd_target.pth'
CHECKPOINT_RND_PREDICTOR = './checkpoint_rnd_predictor.pth'

t_steps = 50
_steps = 0
n_episodes = 1
track = False

for episode in range(n_episodes):

    total_reward = 0
    life = 2

    state = env.reset()

    # while True:    
    for t in range(t_steps):

        # action = env.action_space.sample()
        # action = agent.act( state, EPSILON )

        # 50 frames p/ action
        # actions for very simple movement
        # SIMPLE_MOVEMENT = [
        # 0    ['NOOP'],
        # 1    ['right'],
        # 2    ['right', 'A'],
        # 3    ['right', 'B'],
        # 4    ['right', 'A', 'B'],
        # 6    ['A'], 
        # 7    ['left'],
        # ]            
        
        # action = random.choice([1, 4, 2, 4, 2])
        next_state, reward, done, info = env.step(2)

        print(t)
        time.sleep(.1)

        env.render()

        total_reward += reward

        if done:
            break

        if info['life'] < life:
            total_reward = 0
            life = info['life']

        state = next_state
        t_steps += 1

env.close()
