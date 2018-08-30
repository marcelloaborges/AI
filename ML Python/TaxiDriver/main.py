import gym
from IPython.display import clear_output
from agent import Agent
import random

# 'Acrobot-v1' 3
# 'Taxi-v2'    6

env = gym.make('Taxi-v2')
num_actions = env.env.action_space.n


agent = Agent(num_actions)

for i in range(1000):    
    done = False
    s = env.reset()
    
    verbose = i % 10 == 0

    if verbose:
        clear_output(wait=True)        
        print(f"Episode: {i}")

    while not done:
        a = None
        if random.uniform(0, 1) < 0.1:
            a = env.action_space.sample() # Explore action space
        else:            
            a = agent.play(s)
        
        s_, r, done, info = env.step(a) # take a random action                
        agent.learn(s, a, r, s_, verbose)                
        s = s_        

        verbose = False



total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    s = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        a = agent.play(s)
        s_, r, done, info = env.step(a)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")