from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import torch.optim as optim


from simple_memory import SimpleMemory
# from replay_memory import ReplayMemory
from prioritized_replay_memory import PrioritizedReplayMemory

from noise import OUNoise

from model import ActorModel, CriticModel

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
N_STEP = 8
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 
LR_CRITIC = 3e-5        # learning rate of the critic
WEIGHT_DECAY = 0.995    # L2 weight decay

ADD_NOISE = True
MODE = False

CHECKPOINT_ACTOR = './checkpoint_actor.pth'
CHECKPOINT_CRITIC = './checkpoint_critic.pth'

agent_keys = np.arange( 0, num_agents )

local_memory = SimpleMemory(agent_keys)
# shared_memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)
shared_memory = PrioritizedReplayMemory(BUFFER_SIZE, BATCH_SIZE)
noise = OUNoise(action_size)

actor_model = ActorModel(state_size, action_size).to(DEVICE)
actor_target = ActorModel(state_size, action_size).to(DEVICE)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=LR_ACTOR)

critic_model = CriticModel(state_size, action_size).to(DEVICE)
critic_target = CriticModel(state_size, action_size).to(DEVICE)
critic_optimizer = optim.Adam(critic_model.parameters(), lr=LR_CRITIC)


actor_model.load(CHECKPOINT_ACTOR)
actor_target.load(CHECKPOINT_ACTOR)

critic_model.load(CHECKPOINT_CRITIC)
critic_target.load(CHECKPOINT_CRITIC)


agent = Agent(DEVICE, actor_model, local_memory, shared_memory, noise, GAMMA, N_STEP)

optimizer = Optimizer(DEVICE, 
    actor_model, actor_target, actor_optimizer, 
    critic_model, critic_target, critic_optimizer, 
    shared_memory, 
    GAMMA, TAU)


def maddpg_train():
    n_episodes = 10000
    scores = []
    scores_window = deque(maxlen=100)

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=MODE)[brain_name]     # reset the environment    
        states = env_info.vector_observations                 # get the current state (for each agent)
                
        agent.reset()    

        actor_loss = 0
        critic_loss = 0               

        score = np.zeros(num_agents)                          # initialize the score (for each agent)
        steps = 0

        while True:
            actions = agent.act( states, ADD_NOISE )
            
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished            
                        
            agent.step( 
                agent_keys,
                states,
                actions,
                rewards,
                next_states,
                dones
            )            

            actor_loss, critic_loss = optimizer.learn()

            score += rewards                                   # update the score (for each agent)            

            if np.any(dones):                                  # exit loop if episode finished
                break                                    
            
            states = next_states                               # roll over states to next time step

            steps += 1                                

        scores.append(np.max(score))
        scores_window.append(np.max(score))

        print('Episode: \t{} \tSteps: \t{} \tScore: \t{:.2f} \tMax Score: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, steps, np.max(score), np.max(scores), np.mean(scores_window)))  
        print('Actor loss: \t{:.8f}'.format(actor_loss))
        print('Critic loss: \t{:.8f}'.format(critic_loss))     
        print('')       
        if np.mean(scores_window) >= 2000:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break    
    
    actor_model.checkpoint(CHECKPOINT_ACTOR)
    critic_model.checkpoint(CHECKPOINT_CRITIC)


# train the agent
maddpg_train()


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
        actions = agent.act( states ) 
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