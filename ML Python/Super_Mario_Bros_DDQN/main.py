import os

import torch
import torch.optim as optim

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from model import CNNDDQN
# from memory import Memory
from prioritized_memory import PrioritizedMemory
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
ALPHA = 1
GAMMA = 0.9
TAU = 1e-4
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
LR = 1e-4
EPSILON = 0.

CHECKPOINT = './checkpoint.pth'


model = CNNDDQN( DEVICE, action_size ).to(DEVICE)
target_model = CNNDDQN( DEVICE, action_size ).to(DEVICE)
adam = optim.Adam( model.parameters(), lr=LR )

if os.path.isfile(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT))
    target_model.load_state_dict(torch.load(CHECKPOINT))

# memory = Memory(BUFFER_SIZE, BATCH_SIZE)
memory = PrioritizedMemory(BUFFER_SIZE, BATCH_SIZE)

agent = Agent(DEVICE, model, action_size)
optimizer = Optimizer(DEVICE, memory, model, target_model, adam, 
    ALPHA, GAMMA, TAU, UPDATE_EVERY, BUFFER_SIZE, BATCH_SIZE, LR)


t_steps = 10000
n_episodes = 1000
for episode in range(n_episodes):

    total_reward = 0

    state = env.reset()

    # while True:
    for t in range(t_steps):

        # action = env.action_space.sample()
        action = agent.act( state, EPSILON )

        next_state, reward, done, info = env.step(action)

        loss = 0 # optimizer.step(state, action, reward, next_state, done)

        env.render()

        total_reward += reward

        print('\rEpisode: {} \tT_step: \t{} \tTotal Reward: \t{} \tReward: \t{} \tLoss: \t{:.10f}'.format( episode + 1, t, total_reward, reward, loss ), end='')

        if done:
            break

        state = next_state

    model.checkpoint(CHECKPOINT)

env.close()
