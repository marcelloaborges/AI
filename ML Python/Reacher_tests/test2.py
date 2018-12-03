import numpy as np
from unityagents import UnityEnvironment

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
env_info = env.reset(train_mode=False)[brain_name]    # reset the environment
states = env_info.vector_observations
scores = np.zeros(n_agents)

while True:    
    actions = np.random.randn(n_agents, action_size)
    actions = np.clip(actions, -1, 1)  

    # actions = np.zeros((n_agents, action_size))
    # actions[0] = np.random.randn(1, action_size)
    # actions[0] = np.clip(actions[0], -1, 1)      
    
    env_info = env.step( actions )[brain_name]               # send the action to the environment        
    next_states = env_info.vector_observations               # get the next state        
    rewards = env_info.rewards                               # get the reward        
    dones = env_info.local_done                              # see if episode has finished        
    scores += rewards                                        # update the score
        
    states = next_states                                     # roll over the state to next time step        
        
    if np.any( dones ):                                      # exit loop if episode finished        
        break
    
print('Score: {}'.format(np.mean(scores)))