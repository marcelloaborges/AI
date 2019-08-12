import os
import numpy as np

import torch
import torch.optim as optim

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from cnn import CNN
from ddqn import DDQN
from rnd import RNDTargetModel, RNDPredictorModel

from memory import Memory
# from prioritized_memory import PrioritizedMemory
from agent import Agent
from optimizer import Optimizer

from visdom_utils import VisdomI


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ORIGINAL
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# BLACK
env = gym_super_mario_bros.make('SuperMarioBros-v1')
# PIXEL
# env = gym_super_mario_bros.make('SuperMarioBros-v2')
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

cnn = CNN().to(DEVICE)
model = DDQN( cnn.state_size, action_size ).to(DEVICE)
target = DDQN( cnn.state_size, action_size ).to(DEVICE)
optimizer = optim.Adam( list(model.parameters()) + list(cnn.parameters()), lr=LR, weight_decay=1e-4 )

rnd_target = RNDTargetModel( cnn.state_size ).to(DEVICE)
rnd_predictor = RNDPredictorModel( cnn.state_size + action_size ).to(DEVICE)
rnd_optimizer = optim.Adam( rnd_predictor.parameters(), lr=RND_LR, weight_decay=1e-4 )

if os.path.isfile(CHECKPOINT_MODEL):
    cnn.load_state_dict(torch.load(CHECKPOINT_CNN))
    model.load_state_dict(torch.load(CHECKPOINT_MODEL))
    target.load_state_dict(torch.load(CHECKPOINT_MODEL))
    rnd_target.load_state_dict(torch.load(CHECKPOINT_RND_TARGET))
    rnd_predictor.load_state_dict(torch.load(CHECKPOINT_RND_PREDICTOR))


good_memory = Memory(BUFFER_SIZE, BATCH_SIZE)
bad_memory = Memory(BUFFER_SIZE, BATCH_SIZE)
# good_memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)
# bad_memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)


agent = Agent(DEVICE, cnn, model, action_size)
optimizer = Optimizer(
    DEVICE, 
    good_memory, bad_memory,
    cnn, model, target, optimizer, 
    rnd_target, rnd_predictor, rnd_optimizer,
    ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE)


# vsI = VisdomI()

# t_steps = 500
_steps = 0
n_episodes = 100
track = False

for episode in range(n_episodes):

    total_reward = 0
    life = 2

    state = env.reset()

    while True:
    # for t in range(t_steps):

        # action = env.action_space.sample()
        action = agent.act( state, EPSILON )

        next_state, reward, done, info = env.step(action)

        loss, rnd_loss = optimizer.step(state, action, reward, next_state, done)

        env.render()

        total_reward += reward

        # TRACKING        
        _steps += 1
        # if track:
        #     vsI.track( _steps, loss, 'loss' )        
        #     vsI.track( _steps, rnd_loss, 'rnd_loss' )     
        #     vsI.track( _steps, total_reward, 'total_reward' )
        #     # vsI.track( _steps, f_loss.unsqueeze(0), 'RI' )

        print('\rEpisode: {} \tTotal Reward: \t{} \tReward: \t{} \tLife: \t{} \tLoss: \t{:.5f} \tRND Loss: \t{:.5f}'.format( episode + 1, total_reward, reward, life, loss, rnd_loss ), end='')

        if done:
            break

        if info['life'] < life:
            total_reward = 0
            life = info['life']

        state = next_state

    cnn.checkpoint(CHECKPOINT_CNN)    
    model.checkpoint(CHECKPOINT_MODEL)
    rnd_target.checkpoint(CHECKPOINT_RND_TARGET)
    rnd_predictor.checkpoint(CHECKPOINT_RND_PREDICTOR)

env.close()
