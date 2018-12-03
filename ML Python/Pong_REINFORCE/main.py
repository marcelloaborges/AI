import os
import pong_utils

device = pong_utils.device
# print("using device: ",device)

# render ai gym environment
import gym
import time

# PongDeterministic does not contain random frameskip
# so is faster to train than the vanilla Pong-v4 environment
# env = gym.make('PongDeterministic-v4')

# print("List of available actions: ", env.unwrapped.get_action_meanings())

# we will only use the actions 'RIGHTFIRE' = 4 and 'LEFTFIRE" = 5
# the 'FIRE' part ensures that the game starts again after losing a life
# the actions are hard-coded in pong_utils.py


import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride 
        # (round up if not an integer)

        # 2@80x80 to 4@38x38
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2)
        # 4@38x38 to 8@18x18
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)
        # 16@18x18 to 16@8x8
        self.conv3 = nn.Conv2d(8, 16, kernel_size=4, stride=2)

        self.size=16*8*8 # 1024        

        # 3 fully connected layer
        self.fc1 = nn.Linear(self.size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.sig = nn.Sigmoid()
        
    def forward(self, x):        
    
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten the tensor
        x = x.view(-1, self.size)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)        
        x = self.sig(x)

        return x

# use your own policy!
policy = Policy().to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
import torch.optim as optim
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# # Function Definitions
# Here you will define key functions for training. 
# 
# ## Exercise 2: write your own function for training
# (this is the same as policy_loss except the negative sign)
# 
# ### REINFORCE
# you have two choices (usually it's useful to divide by the time since we've normalized our rewards and the time of each trajectory is fixed)
# 
# 1. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\log(\pi_{\theta'}(a_t|s_t))$
# 2. $\frac{1}{T}\sum^T_t R_{t}^{\rm future}\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}$ where $\theta'=\theta$ and make sure that the no_grad is enabled when performing the division

def surrogate(policy, old_probs, states, actions, rewards,
            discount = 0.995, beta=0.01):

    # using clipped surrogate here for tests

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]

    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    # verify how this normalization works exactly
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

    # convert to tensor
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

    ratio = new_probs / old_probs

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = - ( new_probs * torch.log( old_probs + 1.e-10 ) + ( 1.0 - new_probs ) * torch.log( 1.0 - old_probs + 1.e-10 ) )

    return torch.mean(ratio * rewards + beta * entropy)

# # Training
# We are now ready to train our policy!
# WARNING: make sure to turn on GPU, which also enables multicore processing. It may take up to 45 minutes even with GPU enabled, otherwise it will take much longer!

from parallelEnv import parallelEnv
import numpy as np
# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
episode = 2000
# episode = 800

if __name__ == '__main__':

    # widget bar to display progress
    # get_ipython().system('pip install progressbar')
    import progressbar as pb
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=1, seed=1234)

    discount_rate = .99
    beta = .01
    tmax = 320

    # keep track of progress
    mean_rewards = []

    if os.path.isfile('REINFORCE.policy'):
        policy = torch.load('REINFORCE.policy')

    for e in range(episode):

        # collect trajectories
        old_probs, states, actions, rewards = pong_utils.collect_trajectories(envs, policy, tmax=tmax)
            
        total_rewards = np.sum(rewards, axis=0)

        # this is the SOLUTION!
        # use your own surrogate function
        L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        
        # L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L        
            
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        if (e+1)%20 ==0 :            
            print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)
            
        # update progress widget bar
        timer.update(e+1)
        
    timer.finish()

    # save your policy!
    torch.save(policy, 'REINFORCE.policy')

    plt.plot(mean_rewards)    

    # play game after training!
    env = gym.make('PongDeterministic-v4')
    pong_utils.play(env, policy, time=2000)     

    # load your policy if needed
    # policy = torch.load('REINFORCE.policy')

    # try and test out the solution!
    # policy = torch.load('PPO_solution.policy')

