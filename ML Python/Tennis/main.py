from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch

from replay_buffer import ReplayBuffer
from noise import OUNoise
from actor import Actor
from critic import Critic

import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="../Environments/Tennis_Windows_x86_64/Tennis.exe", no_graphics=False, worker_id=1)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# environment information
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like: ', states[0])


# hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

ADD_NOISE = True
SEED = 2

CHECKPOINT_FOLDER = './'

shared_memory = ReplayBuffer(DEVICE, action_size, BUFFER_SIZE, BATCH_SIZE, SEED)
noise = OUNoise(action_size, 2)

ACTOR_0_KEY = 0
ACTOR_1_KEY = 1

actor_0 = Actor(DEVICE, ACTOR_0_KEY, state_size, action_size, SEED, shared_memory, noise, LR_ACTOR, WEIGHT_DECAY, CHECKPOINT_FOLDER)
actor_1 = Actor(DEVICE, ACTOR_1_KEY, state_size, action_size, SEED, shared_memory, noise, LR_ACTOR, WEIGHT_DECAY, CHECKPOINT_FOLDER)
critic = Critic(DEVICE, state_size, action_size, SEED, GAMMA, TAU, LR_CRITIC, WEIGHT_DECAY, CHECKPOINT_FOLDER)

def maddpg_train():
    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = 3000

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        states = env_info.vector_observations                        # get initial states
        actor_0.reset()                                              # reset the agent noise
        actor_1.reset()                                              # reset the agent noise
                                              
        score = np.zeros(n_agents)
        
        while True:
            action_0 = actor_0.act( states[ACTOR_0_KEY], ADD_NOISE )
            action_1 = actor_1.act( states[ACTOR_1_KEY], ADD_NOISE )
            actions = np.concatenate( (action_0, action_1) )
        
            env_info = env.step( actions  )[brain_name]              # send the action to the environment                            
            next_states = env_info.vector_observations               # get the next state        
            rewards = env_info.rewards                               # get the reward        
            dones = env_info.local_done                              # see if episode has finished        

            actor_0.step(states[ACTOR_0_KEY], action_0, rewards[ACTOR_0_KEY], next_states[ACTOR_0_KEY], dones[ACTOR_0_KEY])
            actor_1.step(states[ACTOR_1_KEY], action_1, rewards[ACTOR_1_KEY], next_states[ACTOR_1_KEY], dones[ACTOR_1_KEY])
            
            critic.step(actor_0, shared_memory)
            critic.step(actor_1, shared_memory)

            score += rewards                                         # update the score
        
            states = next_states                                     # roll over the state to next time step        
                                                        
            if np.any( dones ):                                      # exit loop if episode finished        
                break                                        

        actor_0.checkpoint()
        actor_1.checkpoint()
        critic.checkpoint()

        scores.append(np.max(score))
        scores_window.append(np.max(score))
        
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tMax Score: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.max(score), np.max(scores), np.mean(scores_window)), end="")  
        
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break    

    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()    


# train the agent
# maddpg_train()

# test the trained agents
for episode in range(3):
    env_info = env.reset(train_mode=False)[brain_name]           # reset the environment
    states = env_info.vector_observations                        # get initial states
    score = np.zeros(n_agents)
    
    while True:
        action_0 = actor_0.act(states[ACTOR_0_KEY], add_noise=False)
        action_1 = actor_1.act(states[ACTOR_1_KEY], add_noise=False)
        actions = np.concatenate((action_0, action_1))
        
        env_info = env.step( actions )[brain_name]               # send the action to the environment                            
        next_states = env_info.vector_observations               # get the next state        
        rewards = env_info.rewards                               # get the reward        
        dones = env_info.local_done                              # see if episode has finished        
        
        score += rewards

        states = next_states

        if np.any(dones):                              
            break

    print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.max(score)))  

env.close()