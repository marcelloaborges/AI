import os
from os import system
import numpy as np

import cv2 as cv
from collections import deque

import torch
import torch.optim as optim

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from cnn import CNN
from ddqn import DDQN
from rnd import RNDTargetModel, RNDPredictorModel

from unusual_memory import UnusualMemory
# from memory import Memory
# from prioritized_memory import PrioritizedMemory
from agent import Agent
from optimizer import Optimizer

from visdom_utils import VisdomI


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ORIGINAL
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# BLACK
# env = gym_super_mario_bros.make('SuperMarioBros-v1')
# PIXEL
# env = gym_super_mario_bros.make('SuperMarioBros-v2')
# ONLY FIRST STATE
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

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
TAU = 1e-3
UPDATE_EVERY = 16
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 128
LR = 1e-3
EPSILON = 0.05

RND_LR = 5e-3
RND_OUTPUT_SIZE = 128
RND_UPDATE_EVERY = 32

CHECKPOINT_CNN = './checkpoint_cnn.pth'
CHECKPOINT_MODEL = './checkpoint_model.pth'
CHECKPOINT_RND_TARGET = './checkpoint_rnd_target.pth'
CHECKPOINT_RND_PREDICTOR = './checkpoint_rnd_predictor.pth'

cnn = CNN().to(DEVICE)
model = DDQN( cnn.state_size, action_size ).to(DEVICE)
target = DDQN( cnn.state_size, action_size ).to(DEVICE)
optimizer = optim.Adam( list(model.parameters()) + list(cnn.parameters()), lr=LR, weight_decay=1e-4 )

rnd_target = RNDTargetModel( cnn.state_size ).to(DEVICE)
rnd_predictor = RNDPredictorModel( cnn.state_size ).to(DEVICE)
rnd_optimizer = optim.Adam( rnd_predictor.parameters(), lr=RND_LR, weight_decay=1e-4 )

if os.path.isfile(CHECKPOINT_MODEL):
    cnn.load_state_dict(torch.load(CHECKPOINT_CNN))
    model.load_state_dict(torch.load(CHECKPOINT_MODEL))
    target.load_state_dict(torch.load(CHECKPOINT_MODEL))
    rnd_target.load_state_dict(torch.load(CHECKPOINT_RND_TARGET))
    rnd_predictor.load_state_dict(torch.load(CHECKPOINT_RND_PREDICTOR))


memory = UnusualMemory(BUFFER_SIZE, BATCH_SIZE)
# memory = Memory(BUFFER_SIZE, BATCH_SIZE)
# memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)


agent = Agent(DEVICE, cnn, model, action_size)
optimizer = Optimizer(
    DEVICE, 
    memory,
    cnn, model, target, optimizer, 
    rnd_target, rnd_predictor, rnd_optimizer,
    ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE)


# vsI = VisdomI()

# t_steps = 500
_steps = 0
n_frames = 4
resize_dim = ( 60, 64 ) # 240 x 256 / 4

_steps = 0
n_episodes = 300
track = False

def state_resize_n_to_gray_scale(state, dim):
    t_state = cv.resize( state, dim, interpolation = cv.INTER_AREA )
    t_state = cv.cvtColor( t_state, cv.COLOR_BGR2GRAY )

    return t_state

for episode in range(n_episodes):

    total_reward = 0

    state = deque(maxlen=n_frames)

    t_state = env.reset()
    state.append( state_resize_n_to_gray_scale( t_state, resize_dim ) )

    for frame in range(n_frames - 1):
        action = env.action_space.sample()
        next_state, _, _, _ = env.step(action)

        state.append( state_resize_n_to_gray_scale( next_state, resize_dim ) )

    while True:
    # for t in range(t_steps):

        # action = env.action_space.sample()
        action = agent.act( state, EPSILON )

        next_state, reward, done, info = env.step(action)        

        next_state = state_resize_n_to_gray_scale( next_state, resize_dim)
        t_state = state.copy()
        t_state.append( next_state )

        loss, rnd_loss = optimizer.step(state, action, reward, t_state, done)

        env.render()

        total_reward += reward                
        reward_inverse_dist = memory.rewards_distribution()
        

        if _steps % 100 == 0:
            reward_inverse_dist = dict(sorted(reward_inverse_dist.items(), key=lambda kv: kv[0]))
            rd_str = 'DR => '
            for k, value in reward_inverse_dist.items():
                rd_str += ' {}: {:.3f} |'.format(k, value)
            
            system('cls')
            print('\rE: {} St: {} TR: {} R: {} L: {:.3f} RL: {:.3f} | {}'
                .format( episode + 1, _steps, total_reward, reward, loss, rnd_loss, rd_str ), end='')    


        if done:
            break

        state.append( next_state )
        _steps += 1

    cnn.checkpoint(CHECKPOINT_CNN)    
    model.checkpoint(CHECKPOINT_MODEL)
    rnd_target.checkpoint(CHECKPOINT_RND_TARGET)
    rnd_predictor.checkpoint(CHECKPOINT_RND_PREDICTOR)

env.close()
