from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import torch.optim as optim

from model import A2CModel
from agent import A2CAgent

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
N_STEP = 64
GAMMA = 0.99
GAE_TAU = 0.95
ENTROPY_WEIGHT = 0.01
GRADIENT_CLIP = 0.5
LR = 5e-4

CHECKPOINT_GOALIE = './checkpoint_goalie.pth'
CHECKPOINT_STRIKER = './checkpoint_striker.pth'

# Neural model
goalie_model = A2CModel(goalie_state_size, goalie_action_size).to(DEVICE)
striker_model = A2CModel(striker_state_size, striker_action_size).to(DEVICE)

goalie_model.load(CHECKPOINT_GOALIE)
striker_model.load(CHECKPOINT_STRIKER)

goalie_optimizer = optim.Adam( goalie_model.parameters(), lr=LR )
striker_optimizer = optim.Adam( striker_model.parameters(), lr=LR )
# goalie_optimizer = optim.RMSprop( goalie_model.parameters(), lr=LR, alpha=0.99, eps=1e-5 )
# striker_optimizer = optim.RMSprop( striker_model.parameters(), lr=LR, alpha=0.99, eps=1e-5 )

# Actors and Critics
GOALIE_0_KEY = 0
STRIKER_0_KEY = 0

GOALIE_1_KEY = 1
STRIKER_1_KEY = 1

goalie_0 = A2CAgent( DEVICE, GOALIE_0_KEY, goalie_model, goalie_optimizer, N_STEP, GAMMA, GAE_TAU, ENTROPY_WEIGHT, GRADIENT_CLIP )
striker_0 = A2CAgent( DEVICE, STRIKER_0_KEY, striker_model, striker_optimizer, N_STEP, GAMMA, GAE_TAU, ENTROPY_WEIGHT, GRADIENT_CLIP )

goalie_1 = A2CAgent( DEVICE, GOALIE_1_KEY, goalie_model, goalie_optimizer, N_STEP, GAMMA, GAE_TAU, ENTROPY_WEIGHT, GRADIENT_CLIP )
striker_1 = A2CAgent( DEVICE, STRIKER_1_KEY, striker_model, striker_optimizer, N_STEP, GAMMA, GAE_TAU, ENTROPY_WEIGHT, GRADIENT_CLIP )

def a2c_train():
    n_episodes = 5000
    team_0_window_score = deque(maxlen=100)
    team_1_window_score = deque(maxlen=100)

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)                        # reset the environment    

        goalies_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
        strikers_states = env_info[s_brain_name].vector_observations # get initial state (strikers)

        goalies_scores = np.zeros(n_goalie_agents)                   # initialize the score (goalies)
        strikers_scores = np.zeros(n_striker_agents)                 # initialize the score (strikers) 
        goalie_loss = 0
        striker_loss = 0
        
        while True:            
            # select actions and send to environment
            action_goalie_0 = goalie_0.act( goalies_states[goalie_0.KEY] )
            action_striker_0 = striker_0.act( strikers_states[striker_0.KEY] )

            action_goalie_1 = goalie_1.act( goalies_states[goalie_1.KEY] )
            action_striker_1 = striker_1.act( strikers_states[striker_1.KEY] )
            
            # random            
            # action_goalie_1 = np.asarray( [np.random.randint(goalie_action_size)] )
            # action_striker_1 = np.asarray( [np.random.randint(striker_action_size)] )


            actions_goalies = np.array( (action_goalie_0, action_goalie_1) )                                    
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

            goalie_loss = goalie_0.step( 
                goalies_states[goalie_0.KEY],
                goalies_rewards[goalie_0.KEY],
                done
                )
            goalie_loss = goalie_1.step( 
                goalies_states[goalie_1.KEY],
                goalies_rewards[goalie_1.KEY],
                done
                )
            striker_loss = striker_0.step(                 
                striker_states[striker_0.KEY],    
                strikers_rewards[striker_0.KEY],
                done
                )
            striker_loss = striker_1.step(                 
                striker_states[striker_1.KEY],    
                strikers_rewards[striker_1.KEY],
                done
                )

            # exit loop if episode finished
            if done:
                break  

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states                                                       

        goalie_model.checkpoint(CHECKPOINT_GOALIE)
        striker_model.checkpoint(CHECKPOINT_STRIKER)

        team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
        team_0_window_score.append(1 if team_0_score > 0 else 0)

        team_1_score = goalies_scores[GOALIE_1_KEY] + strikers_scores[STRIKER_1_KEY]
        team_1_window_score.append(1 if team_1_score > 0 else 0)
        
        # print('\rScores from episode {}: {} (goalies), {} (strikers)'.format(episode+1, goalies_scores, strikers_scores), end="")
        print('Episode {} \t Goalie Loss: \t {:.10f} \t Striker Loss: \t {:.10f} \n Red Wins: \t{:.0f} \t Score: \t{:.5f} \n Blue Wins: \t{:.0f} \t Score: \t{:.5f} \n Empates: {:.0f}'
            .format( episode + 1, goalie_loss, striker_loss,
                np.count_nonzero(team_0_window_score), team_0_score, 
                np.count_nonzero(team_1_window_score), team_1_score,
                100 - np.count_nonzero(team_0_window_score) - np.count_nonzero(team_1_window_score)                
            )
        )
        
    # plt.plot(np.arange(1, len(scores)+1), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()    


# train the agent
a2c_train()

# test the trained agents
# team_0_window_score = deque(maxlen=100)
# team_1_window_score = deque(maxlen=100)

# for episode in range(50):                                               # play game for n episodes
#     env_info = env.reset(train_mode=False)                              # reset the environment    
#     goalies_states = env_info[g_brain_name].vector_observations         # get initial state (goalies)
#     strikers_states = env_info[s_brain_name].vector_observations        # get initial state (strikers)

#     goalies_scores = np.zeros(n_goalie_agents)                          # initialize the score (goalies)
#     strikers_scores = np.zeros(n_striker_agents)                        # initialize the score (strikers)
#     while True:
#         # select actions and send to environment
#         action_goalie_0, _ = goalie_0.act( goalies_states[goalie_0.KEY] )
#         action_striker_0, _ = striker_0.act( strikers_states[striker_0.KEY] )

#         action_goalie_1, _ = goalie_1.act( goalies_states[goalie_1.KEY] )                
#         action_striker_1, _ = striker_1.act( strikers_states[striker_1.KEY] )

#         actions_goalies = np.array( (action_goalie_0, action_goalie_1) )                                    
#         actions_strikers = np.array( (action_striker_0, action_striker_1) )

#         actions = dict( zip( [g_brain_name, s_brain_name], [actions_goalies, actions_strikers] ) )

#         env_info = env.step(actions)                       
        
#         # get next states
#         goalies_next_states = env_info[g_brain_name].vector_observations         
#         strikers_next_states = env_info[s_brain_name].vector_observations
        
#         # get reward and update scores
#         goalies_rewards = env_info[g_brain_name].rewards  
#         strikers_rewards = env_info[s_brain_name].rewards
#         goalies_scores += goalies_rewards
#         strikers_scores += strikers_rewards
        
#         # check if episode finished
#         done = np.any(env_info[g_brain_name].local_done)  
        
#         # roll over states to next time step
#         goalies_states = goalies_next_states
#         strikers_states = strikers_next_states                

#         # exit loop if episode finished
#         if done:                                           
#             break
        
#     team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
#     team_0_window_score.append(1 if team_0_score > 0 else 0)

#     team_1_score = goalies_scores[goalie_1.KEY] + strikers_scores[striker_1.KEY]
#     team_1_window_score.append(1 if team_1_score > 0 else 0)
    
#     # print('\rScores from episode {}: {} (goalies), {} (strikers)'.format(episode+1, goalies_scores, strikers_scores), end="")
#     print('Episode {} \n Red Wins: \t{:.0f} \t Score: \t{:.5f} \n Blue Wins: \t{:.0f} \t Score: \t{:.5f} \n Empates: {:.0f}'
#         .format( episode + 1,  
#             np.count_nonzero(team_0_window_score), team_0_score, 
#             np.count_nonzero(team_1_window_score), team_1_score,
#             100 - np.count_nonzero(team_0_window_score) - np.count_nonzero(team_1_window_score)                
#         )
#     )

env.close()