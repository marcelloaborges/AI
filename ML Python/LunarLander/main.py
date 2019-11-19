import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from dqn_agent import Agent
import time

env = gym.make('LunarLander-v2')
agent = Agent(state_size=8, action_size=4, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            env.render(mode='rgb_array') 
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 500.0 or i_episode == n_episodes:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

# train
# scores = dqn(5000)

# load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_50.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_150.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_100.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_200.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_400.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_600.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_800.pth'))
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_1000.pth'))

for i in range(50):
    state = env.reset()    
    while True:
        action = agent.act(state)
        env.render(mode='rgb_array')        
        state, reward, done, _ = env.step(action)
        time.sleep(0.005)
        if done:
            break 
            
env.close()