import sys
import gym
import random
import numpy as np

env = gym.make('Blackjack-v0')

# FACES AND ACES = 10
# ACES = 1
# ACTION = HIT(1) AND STICK(0)
# WIN = 1, DRAW = 0, LOSE = -1

Q_table = {}
picks = {}
rewards = {}

for i in range(500000):    
    done = False    
    s = env.reset()

    while not done:
        if s not in Q_table:
            picks[s] = [0, 0]
            rewards[s] = [0, 0]
            Q_table[s] = [0, 0]
        
        probs = [0.8, 0.2] if s[0] > 18 else [0.2, 0.8]
        a = np.random.choice(2, p=probs)        
        
        s_, r, done, info = env.step(a)
        # print(s_, r)
        
        picks[s][a] += 1
        rewards[s][a] += r        
        Q_table[s][a] = rewards[s][a] / picks[s][a]
        
        s_ = s

    if i % 10000 == 0:
        print("\rTrainning {}".format(i), end="")
        sys.stdout.flush()
        
#TEST
for i in range(10):
    done = False    
    s = env.reset()

    print("Game: ", i)
    episode = []
    while not done:        
        a = np.argmax(Q_table[s])

        s_, r, done, info = env.step(a)
        episode.append((a, s_, r))        

        s_ = s
    
    print(episode)
