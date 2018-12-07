from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import torch.optim as optim

from model import Model

from actor import Actor
from critic import Critic

# import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="../Environments/Soccer_Windows_x86_64/Soccer.exe", no_graphics=False, worker_id=1)

# print the brain names
print(env.brain_names)

# set the goalie brain
g_brain_name = env.brain_names[0]
g_brain = env.brains[g_brain_name]

# set the striker brain
s_brain_name = env.brain_names[1]
s_brain = env.brains[s_brain_name]


# reset the environment
env_info = env.reset(train_mode=True)

# number of agents 
n_goalie_agents = len(env_info[g_brain_name].agents)
print('Number of goalie agents:', n_goalie_agents)
n_striker_agents = len(env_info[s_brain_name].agents)
print('Number of striker agents:', n_striker_agents)

# number of actions
goalie_action_size = g_brain.vector_action_space_size
print('Number of goalie actions:', goalie_action_size)
striker_action_size = s_brain.vector_action_space_size
print('Number of striker actions:', striker_action_size)

# examine the state space 
goalie_states = env_info[g_brain_name].vector_observations
goalie_state_size = goalie_states.shape[1]
print('There are {} goalie agents. Each receives a state with length: {}'.format(goalie_states.shape[0], goalie_state_size))
striker_states = env_info[s_brain_name].vector_observations
striker_state_size = striker_states.shape[1]
print('There are {} striker agents. Each receives a state with length: {}'.format(striker_states.shape[0], striker_state_size))


# hyperparameters
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
N_STEP = 8
EPSILON = 0.1
ENTROPY_WEIGHT = 0.01

LOSS_V = .5			    # v loss coefficient
LOSS_ENTROPY = .01 	    # entropy coefficient

LR = 1e-4               # learning rate of the actor 
WEIGHT_DECAY = 0        # L2 weight decay

CHECKPOINT_GOALIE = './checkpoint_goalie.pth'
CHECKPOINT_STRIKER = './checkpoint_striker.pth'

# Neural model
goalie_model = Model(goalie_state_size, goalie_action_size).to(DEVICE)
goalie_optimizer = optim.Adam(goalie_model.parameters(), lr=LR)

striker_model = Model(striker_state_size, striker_action_size).to(DEVICE)
striker_optimizer = optim.Adam(striker_model.parameters(), lr=LR)

goalie_model.load(CHECKPOINT_GOALIE)
striker_model.load(CHECKPOINT_STRIKER)

# Actors and Critics
GOALIE_0_KEY = 0
STRIKER_0_KEY = 0

GOALIE_1_KEY = 1
STRIKER_1_KEY = 1

goalie_0 = Actor(DEVICE, GOALIE_0_KEY, goalie_model, BATCH_SIZE)
goalie_1 = Actor(DEVICE, GOALIE_1_KEY, goalie_model, BATCH_SIZE)
goalie_critic = Critic(DEVICE, goalie_model, goalie_optimizer, GAMMA, N_STEP, EPSILON, ENTROPY_WEIGHT)

striker_0 = Actor(DEVICE, GOALIE_0_KEY, goalie_model, BATCH_SIZE)
striker_1 = Actor(DEVICE, GOALIE_0_KEY, goalie_model, BATCH_SIZE)
striker_critic = Critic(DEVICE, striker_model, striker_optimizer, GAMMA, N_STEP, EPSILON, ENTROPY_WEIGHT)

def a2c_train():
    n_episodes = 3000

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)                        # reset the environment    

        goalies_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
        strikers_states = env_info[s_brain_name].vector_observations # get initial state (strikers)

        goalies_scores = np.zeros(n_goalie_agents)                   # initialize the score (goalies)
        strikers_scores = np.zeros(n_striker_agents)                 # initialize the score (strikers)
        
        while True:            
            # select actions and send to environment
            action_goalie_0, action_prob_goalie_0 = goalie_0.act( goalies_states[goalie_0.KEY] )
            action_goalie_1, action_prob_goalie_1 = goalie_1.act( goalies_states[goalie_1.KEY] )                
            # action_goalie_1 = np.random.randint(goalie_action_size) 
            actions_goalies = np.array( (action_goalie_0, action_goalie_1) )

            action_striker_0, action_prob_striker_0 = striker_0.act( strikers_states[striker_0.KEY] )
            action_striker_1, action_prob_striker_1 = striker_1.act( strikers_states[striker_1.KEY] )
            # action_striker_1 = np.random.randint(striker_action_size)
            actions_strikers = np.array( (action_striker_0, action_striker_1) )

            actions = dict( zip( [g_brain_name, s_brain_name], [actions_goalies, actions_strikers] ) )

        
            env_info = env.step(actions)                                                
            # get next states
            goalies_next_states = env_info[g_brain_name].vector_observations         
            strikers_next_states = env_info[s_brain_name].vector_observations
            
            # get reward and update scores
            goalies_rewards = env_info[g_brain_name].rewards  
            strikers_rewards = env_info[s_brain_name].rewards
            goalies_scores += goalies_rewards
            strikers_scores += strikers_rewards
            
            # check if episode finished
            done = np.any(env_info[g_brain_name].local_done)    

            # exit loop if episode finished
            if done:
                break              

            goalie_0.step( goalies_states[goalie_0.KEY], action_goalie_0, action_prob_goalie_0, goalies_rewards[goalie_0.KEY], goalies_next_states[goalie_0.KEY] )
            goalie_1.step( goalies_states[goalie_1.KEY], action_goalie_1, action_prob_goalie_1, goalies_rewards[goalie_1.KEY], goalies_next_states[goalie_1.KEY] )
            goalie_critic.step( (goalie_0, goalie_1) )            
                        
            striker_0.step( striker_states[striker_0.KEY], action_striker_0, action_prob_striker_0, strikers_rewards[goalie_0.KEY], strikers_next_states[striker_0.KEY] )
            striker_1.step( striker_states[striker_1.KEY], action_striker_1, action_prob_striker_1, strikers_rewards[goalie_1.KEY], strikers_next_states[striker_1.KEY] )
            striker_critic.step( (striker_0, striker_1) )            

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states                                      

        goalie_model.checkpoint(CHECKPOINT_GOALIE)
        striker_model.checkpoint(CHECKPOINT_STRIKER)
        
        # print('\rScores from episode {}: {} (goalies), {} (strikers)'.format(episode+1, goalies_scores, strikers_scores), end="")
        
    # plt.plot(np.arange(1, len(scores)+1), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()    


# train the agent
a2c_train()

# test the trained agents
for episode in range(50):                                               # play game for n episodes
    env_info = env.reset(train_mode=False)                              # reset the environment    
    goalies_states = env_info[g_brain_name].vector_observations         # get initial state (goalies)
    strikers_states = env_info[s_brain_name].vector_observations        # get initial state (strikers)

    goalies_scores = np.zeros(n_goalie_agents)                          # initialize the score (goalies)
    strikers_scores = np.zeros(n_striker_agents)                        # initialize the score (strikers)
    while True:
        # select actions and send to environment
        action_goalie_0, _ = goalie_0.act( goalies_states[goalie_0.KEY] )
        action_goalie_1, _ = goalie_1.act( goalies_states[goalie_1.KEY] )                
        # action_goalie_1 = np.random.randint(goalie_action_size) 
        actions_goalies = np.array( (action_goalie_0, action_goalie_1) )

        action_striker_0, _ = striker_0.act( strikers_states[striker_0.KEY] )
        action_striker_1, _ = striker_1.act( strikers_states[striker_1.KEY] )
        # action_striker_1 = np.random.randint(striker_action_size)
        actions_strikers = np.array( (action_striker_0, action_striker_1) )

        actions = dict( zip( [g_brain_name, s_brain_name], [actions_goalies, actions_strikers] ) )

        env_info = env.step(actions)                       
        
        # get next states
        goalies_next_states = env_info[g_brain_name].vector_observations         
        strikers_next_states = env_info[s_brain_name].vector_observations
        
        # get reward and update scores
        goalies_rewards = env_info[g_brain_name].rewards  
        strikers_rewards = env_info[s_brain_name].rewards
        goalies_scores += goalies_rewards
        strikers_scores += strikers_rewards
        
        # check if episode finished
        done = np.any(env_info[g_brain_name].local_done)  
        
        # roll over states to next time step
        goalies_states = goalies_next_states
        strikers_states = strikers_next_states
        
        # exit loop if episode finished
        if done:                                           
            break
    print('Scores from episode {}: {} (goalies), {} (strikers)'.format(episode+1, goalies_scores, strikers_scores))

env.close()