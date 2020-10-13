import os
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from torchvision import models, transforms

from PIL import Image

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

from agent_dqn import Agent

from model import AttentionEncoderModel, AttentionActionModel, ActorModel, CriticModel, DQNModel


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


# STATE UTILS
imgToTensor = transforms.ToTensor()
tensorToImg = transforms.ToPILImage()

state_example = env.reset()

# 240 W
# 256 H
height_pixel_cut = 15
img_w = int(state_example.shape[0]/3)
img_h = int(state_example.shape[1]/3)



# HYPERPARAMETERS
EPSILON = 0.1

LR = 1e-5

SEQ_LEN = 24

UPDATE_EVERY = 16
BATCH_SIZE = 128

GAMMA = 0.995
TAU = 1e-3

CHECKPOINT_FOLDER = './'

CHECKPOINT_ATTENTION = CHECKPOINT_FOLDER + 'ATTENTION.pth'
CHECKPOINT_ACTION = CHECKPOINT_FOLDER + 'ACTION.pth'
CHECKPOINT_DQN = CHECKPOINT_FOLDER + 'DQN.pth'

BATCH_FOLDER = CHECKPOINT_FOLDER + 'batch/'

# MODELS
attention_model = AttentionEncoderModel(
        seq_len=SEQ_LEN,       
        attention_heads=4, 
        img_h=img_h - height_pixel_cut,
        img_w=img_w,
        compressed_features_size=128, 
        device=DEVICE
    ).to(DEVICE)

action_model = AttentionActionModel(
        encoding_size=128,
        action_size=action_size,
        fc1_units=64,
        fc2_units=32,
        device=DEVICE
    ).to(DEVICE)

model = DQNModel(
        state_size=128,
        action_size=action_size,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32
    ).to(DEVICE)

target_model = DQNModel(
        state_size=128,
        action_size=action_size,
        fc1_units=256,
        fc2_units=128,
        fc3_units=64,
        fc4_units=32
    ).to(DEVICE)

optimizer = optim.RMSprop( list(attention_model.parameters()) + list(action_model.parameters()), lr=LR )
agent_optimizer = optim.Adam( model.parameters(), lr=LR )

attention_model.load(CHECKPOINT_ATTENTION, DEVICE)
action_model.load(CHECKPOINT_ACTION, DEVICE)
model.load(CHECKPOINT_DQN, DEVICE)
target_model.load(CHECKPOINT_DQN, DEVICE)

# AGENT

agent = Agent(DEVICE,     
    action_size,
    UPDATE_EVERY, BATCH_SIZE, GAMMA, TAU, 
    attention_model, model, target_model, 
    agent_optimizer,
    CHECKPOINT_DQN )

# TRAIN

# EXP COLLECTING

# from data_extractor import DataExtractor
# de = DataExtractor( 
#     DEVICE,
#     env,     
#     action_size, 
#     img_h, img_w, 
#     SEQ_LEN,
#     attention_model, dqn_model,
#     BATCH_SIZE,
#     128000,
#     BATCH_FOLDER
#     )

# # de.extract(height_pixel_cut)

# TRANSFORMER TRAINNING

# from gpt2 import GPT2
# _gpt2 = GPT2( 
#     DEVICE,
#     BATCH_FOLDER,     
#     attention_model, action_model, 
#     optimizer,
#     CHECKPOINT_ATTENTION,
#     CHECKPOINT_ACTION
#     )

# n_epoches = 300
# # _gpt2.train( n_epoches )

# DQN TRAINNING

from dqn import DQN
_dqn = DQN( 
    DEVICE,
    env,
    agent,
    action_size,
    img_h, img_w,
    SEQ_LEN    
 )

n_episodes = 10000
_dqn.train( n_episodes )
