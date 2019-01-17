from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch

from agent import Agent
# import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="../Environments/Soccer_Windows_x86_64/Soccer.exe", no_graphics=True, worker_id=1)

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
N_STEP = 8
BATCH_SIZE = 32
GAMMA = 0.995
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001
GRADIENT_CLIP = 0.5
LR = 8e-5
GOALIE_LR = 5e-5
STRIKER_LR = 1e-4

CHECKPOINT_FOLDER = './'

# Actors and Critics
GOALIE_TYPE = 'GOALIE'
STRIKER_TYPE = 'STRIKER'

GOALIE_0_KEY = 0
STRIKER_0_KEY = 0

GOALIE_1_KEY = 1
STRIKER_1_KEY = 1

goalie_0 = Agent( 
    DEVICE, 
    GOALIE_0_KEY, 
    GOALIE_TYPE, 
    goalie_state_size, 
    goalie_action_size,
    striker_state_size, 
    goalie_state_size,
    striker_state_size, 
    LR,
    N_STEP,    
    BATCH_SIZE,
    GAMMA,
    EPSILON,
    ENTROPY_WEIGHT,
    GRADIENT_CLIP,
    CHECKPOINT_FOLDER
    )
goalie_1 = Agent( 
    DEVICE, 
    GOALIE_1_KEY, 
    GOALIE_TYPE, 
    goalie_state_size, 
    goalie_action_size,
    striker_state_size, 
    goalie_state_size,
    striker_state_size, 
    LR,
    N_STEP,
    BATCH_SIZE,
    GAMMA,
    EPSILON,
    ENTROPY_WEIGHT,
    GRADIENT_CLIP,
    CHECKPOINT_FOLDER   
    )

striker_0 = Agent( 
    DEVICE, 
    STRIKER_0_KEY, 
    STRIKER_TYPE, 
    striker_state_size, 
    striker_action_size,
    goalie_state_size, 
    striker_state_size,
    goalie_state_size, 
    LR,
    N_STEP,
    BATCH_SIZE,
    GAMMA,
    EPSILON,
    ENTROPY_WEIGHT,
    GRADIENT_CLIP,
    CHECKPOINT_FOLDER
    )
striker_1 = Agent( 
    DEVICE, 
    STRIKER_1_KEY, 
    STRIKER_TYPE, 
    striker_state_size, 
    striker_action_size,
    goalie_state_size, 
    striker_state_size,
    goalie_state_size, 
    LR,
    N_STEP,
    BATCH_SIZE,
    GAMMA,
    EPSILON,
    ENTROPY_WEIGHT,
    GRADIENT_CLIP,
    CHECKPOINT_FOLDER
    )

def ppo_train():
    n_episodes = 100000
    team_0_window_score = deque(maxlen=100)
    team_0_window_score_wins = deque(maxlen=100)

    team_1_window_score = deque(maxlen=100)
    team_1_window_score_wins = deque(maxlen=100)

    draws = deque(maxlen=100)

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)                        # reset the environment    

        goalies_states = env_info[g_brain_name].vector_observations  # get initial state (goalies)
        strikers_states = env_info[s_brain_name].vector_observations # get initial state (strikers)

        goalies_scores = np.zeros(n_goalie_agents)                   # initialize the score (goalies)
        strikers_scores = np.zeros(n_striker_agents)                 # initialize the score (strikers)         

        steps = 0
        
        while True:       
            # select actions and send to environment
            action_goalie_0, log_prob_goalie_0 = goalie_0.act( goalies_states[goalie_0.KEY] )
            action_striker_0, log_prob_striker_0 = striker_0.act( strikers_states[striker_0.KEY] )

            # action_goalie_1, log_prob_goalie_1 = goalie_1.act( goalies_states[goalie_1.KEY] )
            # action_striker_1, log_prob_striker_1 = striker_1.act( strikers_states[striker_1.KEY] )
            
            # random            
            action_goalie_1 = np.asarray( [np.random.randint(goalie_action_size)] )
            action_striker_1 = np.asarray( [np.random.randint(striker_action_size)] )


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

            # learn
            goalie_0_reward = goalies_rewards[goalie_0.KEY] # * 0.6 + strikers_rewards[striker_0.KEY] * 0.4
            goalie_0.step( 
                goalies_states[goalie_0.KEY],
                striker_states[striker_0.KEY],
                goalies_states[GOALIE_1_KEY], 
                striker_states[STRIKER_1_KEY],
                action_goalie_0,
                log_prob_goalie_0,
                goalie_0_reward 
                )
            # goalie_1_reward = goalies_rewards[goalie_1.KEY] * 0.6 + strikers_rewards[striker_1.KEY] * 0.4     

            striker_0_reward = strikers_rewards[striker_0.KEY] # * 0.6 + goalies_rewards[goalie_0.KEY] * 0.4
            striker_0.step(                 
                striker_states[striker_0.KEY],
                goalies_states[goalie_0.KEY],
                striker_states[STRIKER_1_KEY],
                goalies_states[GOALIE_1_KEY],                
                action_striker_0,
                log_prob_striker_0,
                striker_0_reward
                )
            # striker_1_reward = strikers_rewards[striker_1.KEY] * 0.6 + goalies_rewards[goalie_1.KEY] * 0.4    

            # exit loop if episode finished
            if done:
                break  

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states

            steps += 1

        goalie_loss = goalie_0.optimize()
        striker_loss = striker_0.optimize()

        goalie_0.checkpoint()
        striker_0.checkpoint()

        team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
        team_0_window_score.append( team_0_score )
        team_0_window_score_wins.append( 1 if team_0_score > 0 else 0)        

        team_1_score = goalies_scores[GOALIE_1_KEY] + strikers_scores[STRIKER_1_KEY]                
        team_1_window_score.append( team_1_score )
        team_1_window_score_wins.append( 1 if team_1_score > 0 else 0 )

        draws.append( team_0_score == team_1_score )
        
        print('Episode: {} \tSteps: \t{} \tGoalie Loss: \t {:.10f} \tStriker Loss: \t {:.10f}'.format( episode + 1, steps, goalie_loss, striker_loss ))
        print('\tRed Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_0_window_score_wins), team_0_score, np.sum(team_0_window_score) ))
        print('\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_1_window_score_wins), team_1_score, np.sum(team_0_window_score) ))
        print('\tDraws: \t{}'.format( np.count_nonzero(draws) ))

        if np.count_nonzero(team_0_window_score) >= 100:
            break
        
    # plt.plot(np.arange(1, len(scores)+1), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()    


# train the agent
ppo_train()

# test the trained agents
team_0_window_score = deque(maxlen=100)
team_0_window_score_wins = deque(maxlen=100)

team_1_window_score = deque(maxlen=100)
team_1_window_score_wins = deque(maxlen=100)

draws = deque(maxlen=100)

for episode in range(50):                                               # play game for n episodes
    env_info = env.reset(train_mode=False)                              # reset the environment    
    goalies_states = env_info[g_brain_name].vector_observations         # get initial state (goalies)
    strikers_states = env_info[s_brain_name].vector_observations        # get initial state (strikers)

    goalies_scores = np.zeros(n_goalie_agents)                          # initialize the score (goalies)
    strikers_scores = np.zeros(n_striker_agents)                        # initialize the score (strikers)

    steps = 0

    while True:
        # select actions and send to environment
            action_goalie_0, log_prob_goalie_0 = goalie_0.act( goalies_states[goalie_0.KEY] )
            action_striker_0, log_prob_striker_0 = striker_0.act( strikers_states[striker_0.KEY] )

            # action_goalie_1, log_prob_goalie_1 = goalie_1.act( goalies_states[goalie_1.KEY] )
            # action_striker_1, log_prob_striker_1 = striker_1.act( strikers_states[striker_1.KEY] )
            
            # random            
            action_goalie_1 = np.asarray( [np.random.randint(goalie_action_size)] )
            action_striker_1 = np.asarray( [np.random.randint(striker_action_size)] )


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

            # exit loop if episode finished
            if done:
                break  

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states

            steps += 1
        
    team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
    team_0_window_score.append( team_0_score )
    team_0_window_score_wins.append( 1 if team_0_score > 0 else 0)        

    team_1_score = goalies_scores[GOALIE_1_KEY] + strikers_scores[STRIKER_1_KEY]                
    team_1_window_score.append( team_1_score )
    team_1_window_score_wins.append( 1 if team_1_score > 0 else 0 )

    draws.append( team_0_score == team_1_score )
    
    print('Episode {}'.format( episode + 1 ))
    print('\tRed Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_0_window_score_wins), team_0_score, np.sum(team_0_window_score) ))
    print('\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}'.format( np.count_nonzero(team_1_window_score_wins), team_1_score, np.sum(team_0_window_score) ))
    print('\tDraws: \t{}'.format( np.count_nonzero( draws ) ))

env.close()