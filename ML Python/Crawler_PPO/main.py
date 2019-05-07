from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import torch.optim as optim

from model import ActorCriticModel
from simple_memory import SimpleMemory
from agent import Agent
from optimizer import Optimizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="../Environments/Crawler_Windows_x86_64/Crawler.exe", no_graphics=False)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# hyperparameters
N_STEP = 128
GAMMA = 0.99            # discount factor
BATCH_SIZE = 32
EPSILON = 0.1
ENTROPY_WEIGHT = 0.001
GRADIENT_CLIP = 0.5
LR = 3e-4               # learning rate of the critic
WEIGHT_DECAY = 0.995    # L2 weight decay

MODE = True

CHECKPOINT = './checkpoint.pth'

agent_keys = np.arange( 0, num_agents )

shared_memory = SimpleMemory(agent_keys)

actor_critic_model = ActorCriticModel(state_size, action_size).to(DEVICE)
actor_critic_optimizer = optim.Adam(actor_critic_model.parameters(), lr=LR)

actor_critic_model.load(CHECKPOINT)

agent = Agent(DEVICE, actor_critic_model, shared_memory)

optimizer = Optimizer(DEVICE, 
    actor_critic_model, actor_critic_optimizer, 
    shared_memory, 
    N_STEP, GAMMA, BATCH_SIZE, EPSILON, ENTROPY_WEIGHT, GRADIENT_CLIP)


def ppo_train():
    n_episodes = 10000
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=MODE)[brain_name]     # reset the environment    
        states = env_info.vector_observations                 # get the current state (for each agent)
                
        loss = 0

        score = np.zeros(num_agents)                          # initialize the score (for each agent)
        steps = 0

        while True:
            actions, log_probs, values = agent.act( states )
            
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished            
                        
            agent.step( 
                agent_keys,
                states,
                actions,
                log_probs,
                rewards,
                values
            )            

            loss = optimizer.step()

            score += rewards                                   # update the score (for each agent)            

            if np.any(dones):                                  # exit loop if episode finished
                break                                    
            
            states = next_states                               # roll over states to next time step

            steps += 1                                

        scores.append(np.max(score))
        scores_window.append(np.max(score))

        print('Episode: \t{} \tSteps: \t{} \tScore: \t{:.2f} \tMax Score: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, steps, np.max(score), np.max(scores), np.mean(scores_window)))  
        print('Loss: \t{:.8f}'.format(loss))
        print('')       
        if np.mean(scores_window) >= 2000:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break    
    
    actor_critic_model.checkpoint(CHECKPOINT)


# train the agent
ppo_train()


n_episodes = 50
scores = []
scores_window = deque(maxlen=100)

for episode in range(n_episodes):
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                 # get the current state (for each agent)      

    score = np.zeros(num_agents)                          # initialize the score (for each agent)
    steps = 0

    while True:
        # n_actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        # n_actions = np.clip(n_actions, -1, 1)                  # all actions between -1 and 1
        actions = []
        actions, _, _ = agent.act( states ) 
        # actions = np.stack( actions, axis=0 )

        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished                

        score += rewards                                   # update the score (for each agent)

        states = next_states                               # roll over states to next time step

        if np.any(dones):                                  # exit loop if episode finished
            break
        
        steps += 1

    scores.append(np.max(score))
    scores_window.append(np.max(score))

    print('\rEpisode: \t{} \tScore: \t{:.2f} \tMax Score: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.max(score), np.max(scores), np.mean(scores_window)), end="")  

env.close()