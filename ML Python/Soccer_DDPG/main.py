from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import torch.optim as optim

from model import ActorModel, CriticModel
from replay_buffer import ReplayBuffer
from noise import OUNoise

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
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

ADD_NOISE = True
SEED = 2

CHECKPOINT_GOALIE = './checkpoint_goalie.pth'
CHECKPOINT_STRIKER = './checkpoint_striker.pth'
CHECKPOINT_GOALIE_CRITIC = './checkpoint_goalie_critic.pth'
CHECKPOINT_STRIKER_CRITIC = './checkpoint_striker_critic.pth'

# Shared memory and Noise
goalie_noise = OUNoise(goalie_action_size, 2)
goalie_memory = ReplayBuffer(DEVICE, goalie_action_size, BUFFER_SIZE, BATCH_SIZE, SEED)

striker_noise = OUNoise(striker_action_size, 2)
striker_memory = ReplayBuffer(DEVICE, striker_action_size, BUFFER_SIZE, BATCH_SIZE, SEED)

# Actors and Critics
GOALIE_0_KEY = 0
STRIKER_0_KEY = 0

GOALIE_1_KEY = 1
STRIKER_1_KEY = 1

# Actor Network (w/ Target Network)
goalie_local_model = ActorModel(goalie_state_size, goalie_action_size, 2).to(DEVICE)
goalie_target_model = ActorModel(goalie_state_size, goalie_action_size, 2).to(DEVICE)
goalie_optimizer = optim.Adam(goalie_local_model.parameters(), lr=LR_ACTOR)

striker_local_model = ActorModel(striker_state_size, striker_action_size, 2).to(DEVICE)
striker_target_model = ActorModel(striker_state_size, striker_action_size, 2).to(DEVICE)
striker_optimizer = optim.Adam(striker_local_model.parameters(), lr=LR_ACTOR)

# Critic Network (w/ Target Network)
critic_goalie_local = CriticModel(goalie_state_size, goalie_action_size, 2).to(DEVICE)
critic_goalie_target = CriticModel(goalie_state_size, goalie_action_size, 2).to(DEVICE)
critic_goalie_optimizer = optim.Adam(critic_goalie_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

critic_striker_local = CriticModel(striker_state_size, striker_action_size, 2).to(DEVICE)
critic_striker_target = CriticModel(striker_state_size, striker_action_size, 2).to(DEVICE)
critic_striker_optimizer = optim.Adam(critic_striker_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)



goalie_0 = Actor(DEVICE, GOALIE_0_KEY, goalie_local_model, goalie_target_model, goalie_optimizer, goalie_memory, goalie_noise, CHECKPOINT_GOALIE)
striker_0 = Actor(DEVICE, STRIKER_0_KEY, striker_local_model, striker_target_model, striker_optimizer, striker_memory, striker_noise, CHECKPOINT_STRIKER)

goalie_1 = Actor(DEVICE, GOALIE_1_KEY, goalie_local_model, goalie_target_model, goalie_optimizer, goalie_memory, goalie_noise, CHECKPOINT_GOALIE)
striker_1 = Actor(DEVICE, STRIKER_1_KEY, striker_local_model, striker_target_model, striker_optimizer, striker_memory, striker_noise, CHECKPOINT_STRIKER)

critic_goalie = Critic(DEVICE, critic_goalie_local, critic_goalie_target, critic_goalie_optimizer, GAMMA, TAU, CHECKPOINT_GOALIE_CRITIC)
critic_striker = Critic(DEVICE, critic_striker_local, critic_striker_target, critic_striker_optimizer, GAMMA, TAU, CHECKPOINT_STRIKER_CRITIC)

def a2c_train():
    n_episodes = 10000
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
            action_goalie_0, prob_goalie_0 = goalie_0.act( goalies_states[goalie_0.KEY] )
            action_striker_0, prob_striker_0 = striker_0.act( strikers_states[striker_0.KEY] )

            action_goalie_1, prob_goalie_1 = goalie_1.act( goalies_states[goalie_1.KEY] )                
            action_striker_1, prob_striker_1 = striker_1.act( strikers_states[striker_1.KEY] )

            # random
            # action_goalie_0 = np.random.randint(goalie_action_size) 
            # action_striker_0 = np.random.randint(striker_action_size)
            
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

            # goalie_0_reward = goalies_rewards[goalie_0.KEY] * ( 1 - 0.4) + 2 * 0.4
            # striker_0_reward = strikers_rewards[striker_0.KEY] * ( 1 - 0.4) + 2 * 0.4
            # team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]            

            goalie_0.step(goalies_states[GOALIE_0_KEY], prob_goalie_0, goalies_rewards[GOALIE_0_KEY], goalies_next_states[GOALIE_0_KEY], done)
            striker_0.step(striker_states[STRIKER_0_KEY], prob_striker_0, strikers_rewards[STRIKER_0_KEY], strikers_next_states[STRIKER_0_KEY], done)
            
            goalie_1.step(goalies_states[GOALIE_1_KEY], prob_goalie_1, goalies_rewards[GOALIE_1_KEY], goalies_next_states[GOALIE_1_KEY], done)
            striker_1.step(striker_states[STRIKER_1_KEY], prob_striker_1, strikers_rewards[STRIKER_1_KEY], strikers_next_states[STRIKER_1_KEY], done)
            
            critic_goalie.step(goalie_local_model, goalie_target_model, goalie_optimizer, goalie_memory)
            critic_striker.step(striker_local_model, striker_target_model, striker_optimizer, striker_memory)

            # exit loop if episode finished
            if done:
                break  

            # roll over states to next time step
            goalies_states = goalies_next_states
            strikers_states = strikers_next_states                                                       

        goalie_0.checkpoint()
        striker_0.checkpoint()
        critic_goalie.checkpoint()
        critic_striker.checkpoint()

        team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
        team_0_window_score.append(1 if team_0_score > 0 else 0)

        team_1_score = goalies_scores[goalie_1.KEY] + strikers_scores[striker_1.KEY]
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
team_0_window_score = deque(maxlen=100)
team_1_window_score = deque(maxlen=100)

for episode in range(50):                                               # play game for n episodes
    env_info = env.reset(train_mode=False)                              # reset the environment    
    goalies_states = env_info[g_brain_name].vector_observations         # get initial state (goalies)
    strikers_states = env_info[s_brain_name].vector_observations        # get initial state (strikers)

    goalies_scores = np.zeros(n_goalie_agents)                          # initialize the score (goalies)
    strikers_scores = np.zeros(n_striker_agents)                        # initialize the score (strikers)

    goalie_0.reset()                                              # reset the agent noise
    goalie_1.reset()                                              # reset the agent noise

    striker_0.reset()                                              # reset the agent noise
    striker_1.reset()                                              # reset the agent noise

    while True:
        # select actions and send to environment
        action_goalie_0, _ = goalie_0.act( goalies_states[goalie_0.KEY] )
        action_striker_0, _ = striker_0.act( strikers_states[striker_0.KEY] )

        action_goalie_1, _ = goalie_1.act( goalies_states[goalie_1.KEY] )                
        action_striker_1, _ = striker_1.act( strikers_states[striker_1.KEY] )

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
        
        # roll over states to next time step
        goalies_states = goalies_next_states
        strikers_states = strikers_next_states                

        # exit loop if episode finished
        if done:                                           
            break
        
    team_0_score = goalies_scores[goalie_0.KEY] + strikers_scores[striker_0.KEY]
    team_0_window_score.append(1 if team_0_score > 0 else 0)

    team_1_score = goalies_scores[goalie_1.KEY] + strikers_scores[striker_1.KEY]
    team_1_window_score.append(1 if team_1_score > 0 else 0)
    
    # print('\rScores from episode {}: {} (goalies), {} (strikers)'.format(episode+1, goalies_scores, strikers_scores), end="")
    print('Episode {} \n Red Wins: \t{:.0f} \t Score: \t{:.5f} \n Blue Wins: \t{:.0f} \t Score: \t{:.5f} \n Empates: {:.0f}'
        .format( episode + 1,  
            np.count_nonzero(team_0_window_score), team_0_score, 
            np.count_nonzero(team_1_window_score), team_1_score,
            100 - np.count_nonzero(team_0_window_score) - np.count_nonzero(team_1_window_score)                
        )
    )

env.close()