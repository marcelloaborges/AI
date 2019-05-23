import torch
import torch.optim as optim

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from model import CNNActorCriticModel
from memory import Memory
from agent import Agent
from optimizer import Optimizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

state_info = env.reset()
action_info = env.action_space.sample()

print('states looks like {}'.format(state_info))

print('states len {}'.format(state_info.shape))
print('actions len {}'.format(env.action_space))


# hyperparameters
N_STEP = 64
LR = 1e-4

BATCH_SIZE = 32
GAMMA = 0.9
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001
GRADIENT_CLIP = 1

CHECKPOINT = './checkpoint.pth'


model = CNNActorCriticModel( env.action_space.n ).to(DEVICE)
adam = optim.Adam( model.parameters(), lr=LR )
memory = Memory()
agent = Agent(DEVICE, model)
optimizer = Optimizer(DEVICE, memory, model, adam, 
    N_STEP, BATCH_SIZE, GAMMA, EPSILON, ENTROPY_WEIGHT, GRADIENT_CLIP)


n_episodes = 10
for episode in range(n_episodes):

    state = env.reset()

    while True:

        # action = env.action_space.sample()
        action, log_prob = agent.act( state )

        next_state, reward, done, info = env.step(action)

        loss = optimizer.step(state, action, log_prob, reward)

        env.render()

        print('\rEpisode: {} \tReward: \t {} \tLoss: \t {:.10f}'.format( episode + 1, reward, loss ), end='')

        if done:
            break

        state = next_state

    model.checkpoint(CHECKPOINT)

    env.close()
