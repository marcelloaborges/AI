import os
import numpy as np

import torch
import torch.optim as optim

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from cnn import CNN
from d4pg import D4PGActor, D4PGCritic
from rnd import RNDTargetModel, RNDPredictorModel

from noise import OUNoise

from memory import Memory
from agent import Agent
from optimizer import Optimizer


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
GAMMA = 0.99
TAU = 2e-1
UPDATE_EVERY = 16
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

ADD_NOISE = True

RND_LR = 5e-5
RND_OUTPUT_SIZE = 128
RND_UPDATE_EVERY = 32

CHECKPOINT_CNN = './checkpoint_cnn.pth'
CHECKPOINT_ACTOR = './checkpoint_actor.pth'
CHECKPOINT_CRITIC = './checkpoint_critic.pth'
CHECKPOINT_RND_TARGET = './checkpoint_rnd_target.pth'
CHECKPOINT_RND_PREDICTOR = './checkpoint_rnd_predictor.pth'

cnn = CNN().to(DEVICE)

actor_model = D4PGActor( cnn.state_size, action_size ).to(DEVICE)
actor_target = D4PGActor( cnn.state_size, action_size ).to(DEVICE)

critic_model = D4PGCritic( cnn.state_size, action_size ).to(DEVICE)
critic_target = D4PGCritic( cnn.state_size, action_size ).to(DEVICE)

optimizer_actor = optim.Adam( list(actor_model.parameters()) + list(cnn.parameters()), lr=LR_ACTOR, weight_decay=1e-4 )
optimizer_critic = optim.Adam( critic_model.parameters(), lr=LR_CRITIC, weight_decay=1e-4 )

rnd_target = RNDTargetModel( cnn.state_size ).to(DEVICE)
rnd_predictor = RNDPredictorModel( cnn.state_size + action_size ).to(DEVICE)
rnd_optimizer = optim.Adam( rnd_predictor.parameters(), lr=RND_LR, weight_decay=1e-4 )

if os.path.isfile(CHECKPOINT_CNN):
    cnn.load_state_dict(torch.load(CHECKPOINT_CNN))
    
    actor_model.load_state_dict(torch.load(CHECKPOINT_ACTOR))
    actor_target.load_state_dict(torch.load(CHECKPOINT_ACTOR))

    critic_model.load_state_dict(torch.load(CHECKPOINT_CRITIC))
    critic_target.load_state_dict(torch.load(CHECKPOINT_CRITIC))

    rnd_target.load_state_dict(torch.load(CHECKPOINT_RND_TARGET))
    rnd_predictor.load_state_dict(torch.load(CHECKPOINT_RND_PREDICTOR))


good_memory = Memory(BUFFER_SIZE, BATCH_SIZE)
bad_memory = Memory(BUFFER_SIZE, BATCH_SIZE)
# good_memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)
# bad_memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)

noise = OUNoise(action_size)

agent = Agent(DEVICE, cnn, actor_model, noise)
optimizer = Optimizer(
    DEVICE, 
    good_memory, bad_memory,
    cnn, 
    actor_model, actor_target, optimizer_actor, 
    critic_model, critic_target, optimizer_critic, 
    rnd_target, rnd_predictor, rnd_optimizer,
    ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE)


# t_steps = 500
n_episodes = 100

for episode in range(n_episodes):

    total_reward = 0
    life = 2

    state = env.reset()

    while True:
    # for t in range(t_steps):

        # action = env.action_space.sample()
        action = agent.act( state, ADD_NOISE )

        next_state, reward, done, info = env.step(action)

        actor_loss, critic_loss, rnd_loss = optimizer.step(state, action, reward, next_state, done)

        env.render()

        total_reward += reward

        print('\rEpisode: {} \tTotal: \t{} \tReward: \t{} \tLife: \t{} \tActor: \t{:.5f} \tCritic: \t{:.5f} \tRND: \t{:.5f}'.format( episode + 1, total_reward, reward, life, actor_loss, critic_loss, rnd_loss ), end='')

        if done:
            break

        if info['life'] < life:
            total_reward = 0
            life = info['life']

        state = next_state

    cnn.checkpoint(CHECKPOINT_CNN)    
    
    actor_model.checkpoint(CHECKPOINT_ACTOR)
    critic_model.checkpoint(CHECKPOINT_CRITIC)

    rnd_target.checkpoint(CHECKPOINT_RND_TARGET)
    rnd_predictor.checkpoint(CHECKPOINT_RND_PREDICTOR)

env.close()
