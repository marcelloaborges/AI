import os
import numpy as np

import torch
import torch.optim as optim

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from model import CNNLSTMActorCriticModel
from memory import Memory
from agent import Agent
from optimizer import Optimizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

state_info = env.reset()
action_info = env.action_space.sample()
action_size = env.action_space.n

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(action_size))


# hyperparameters
N_STEP = 256
LR = 1e-4

BATCH_SIZE = 32
GAMMA = 0.9
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001
GRADIENT_CLIP = 1

LSTM_HIDDEN_UNITS = 512
FC1_UNITS = 256

CHECKPOINT = './checkpoint.pth'


model = CNNLSTMActorCriticModel( action_size, lstm_hidden_units=LSTM_HIDDEN_UNITS, fc1_units=FC1_UNITS ).to(DEVICE)
adam = optim.Adam( model.parameters(), lr=LR )

if os.path.isfile(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT))

memory = Memory()

agent = Agent(DEVICE, model)
optimizer = Optimizer(DEVICE, memory, model, adam, 
    N_STEP, BATCH_SIZE, GAMMA, EPSILON, ENTROPY_WEIGHT, GRADIENT_CLIP)


t_steps = 10000
n_episodes = 1000
for episode in range(n_episodes):

    total_reward = 0
    t_step = 0
    life = 2

    state = env.reset()
    hx = np.zeros( [1, 1, LSTM_HIDDEN_UNITS] )
    cx = np.zeros( [1, 1, LSTM_HIDDEN_UNITS] )

    for _ in range(t_steps):
        t_step += 1    

        # action = env.action_space.sample()
        action, log_prob, nhx, ncx = agent.act( state, hx, cx )

        next_state, reward, done, info = env.step(action)

        loss = 0 # optimizer.step(state, hx, cx, action, log_prob, reward)

        env.render()
        
        total_reward += reward

        print('\rEpisode: {} \tT_step: \t{} \tTotal Reward: \t{} \tReward: \t{} \tLife: \t{} \tLoss: \t{:.10f}'.format( episode + 1, t_step, total_reward, reward, life, loss ), end='')

        if done:
            break

        if info['life'] < life:
            total_reward = 0
            t_step = 0
            life = info['life']

        state = next_state
        hx = nhx
        cx = ncx

    model.checkpoint(CHECKPOINT)

env.close()
