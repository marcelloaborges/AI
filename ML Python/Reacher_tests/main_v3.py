import numpy as np
from unityagents import UnityEnvironment

import torch

from v3_DDPG.ddpg_agent import Agent

from collections import deque

# env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe", no_graphics=False)
env = UnityEnvironment(file_name="Reacher_Windows_x86_64 (20)/Reacher.exe", no_graphics=False)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
print('Number of agents:', len(env_info.agents))
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


n_agents = 20
n_episodes = 1000
scores = deque(maxlen=100)

agent = Agent(state_size, action_size, 4)

for episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]    # reset the environment
    states = env_info.vector_observations
    agent.reset()
    score = np.zeros(n_agents)

    while True:    
        # actions = np.random.randn(n_agents, action_size)    
        # actions = np.clip(actions, -1, 1)    
        actions = agent.act(states)
        
        env_info = env.step( actions )[brain_name]               # send the action to the environment        
        next_states = env_info.vector_observations               # get the next state        
        rewards = env_info.rewards                               # get the reward        
        dones = env_info.local_done                              # see if episode has finished        

        actor_loss, critic_loss = agent.step(states, actions, rewards, next_states, dones)

        score += rewards                                         # update the score
             
        states = next_states                                     # roll over the state to next time step        
                            
        print('\rEpisode: \t{} \tEpi score: \t{:.2f} \tMean Score: \t{:.2f} \tActor loss: \t{:.5f} \tCritic loss \t{:.5f}'
                .format(episode, np.mean(score), np.mean(scores), actor_loss, critic_loss), end="")  
                
        if np.any( dones ):                                      # exit loop if episode finished        
            scores.append(score)                 

            torch.save(agent.actor_local.state_dict(), 'v3_DDPG/checkpoint_actor.pth')      
            torch.save(agent.critic_local.state_dict(), 'v3_DDPG/checkpoint_critic.pth')               

            break   
    
    if np.mean(scores) >= 30:
        break          

print(scores)