import numpy as np
import torch
import torch.optim as optim
import os

from unityagents import UnityEnvironment

import logging

from v2_A3C.model import AC_Model
from v2_A3C.brain import Brain
from v2_A3C.manager import Manager
from v2_A3C.environment import Environment
from v2_A3C.agent import Agent
from v2_A3C.work_queue import WorkQueue
from v2_A3C.optimizer import Optimizer

# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s',)


# States have length: 33
# Number of actions: 4
ENV_EXE_URL = "Reacher_Windows_x86_64/Reacher.exe"
STATE_SIZE = 33
ACTION_SIZE = 4
N_EPISODES = 2

# HYPER PARAMETERS
RUN_TIME = 30
N_AGENTS = 1
N_OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

# EPS_START = 0.4
# EPS_STOP  = .15
# EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

# LOSS_V = .5			# v loss coefficient
# LOSS_ENTROPY = .01 	# entropy coefficient


# THE MODE FOR A3C
# IN A3C IT'S SHARED/GLOBAL
model = AC_Model(STATE_SIZE, ACTION_SIZE).to(DEVICE)
brain = Brain(DEVICE, model, LEARNING_RATE, GAMMA, MIN_BATCH)

work_queue = WorkQueue(N_EPISODES)

# THE MAIN LOOP EXECUTED AS A THREAD
ENV = UnityEnvironment(file_name=ENV_EXE_URL, no_graphics=True)
manager_threads = [ Manager(ENV, i, brain, GAMMA, GAMMA_N, N_STEP_RETURN, args=work_queue) for i in range(N_AGENTS) ]
# LEARNERS RUNNING IN PARALEL WITH THE AGENTS
opts = [ Optimizer(brain, args=work_queue) for i in range(N_OPTIMIZERS) ]

for m in manager_threads:
    m.start()

for o in opts:
    o.start()

print("Training started")