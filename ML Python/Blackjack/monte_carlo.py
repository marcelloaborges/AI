import sys
import gym
import numpy as np
from collections import defaultdict

# from plotutils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        ## TODO: complete the function
        episode = generate_episode(env)
        s, a, r = zip(*episode)

        discounts = np.array([gamma**i for i in range(len(r) + 1)])

        for i, state in enumerate(s):
            returns_sum[state][a[i]] += sum(r[i:] * discounts[:-(1 + i)])
            N[state][a[i]] += 1.0
            Q[state][a[i]] = returns_sum[state][a[i]] / N[state][a[i]]
        
    return Q

# obtain the action-value function
Q = mc_prediction_q(env, 100000, generate_episode_from_limit_stochastic)

#TEST
for i in range(10):
    done = False    
    s = env.reset()

    print("Game: ", i)
    episode = []
    while not done:        
        a = np.argmax(Q[s])

        s_, r, done, info = env.step(a)
        episode.append((a, s_, r))        

        s_ = s
    
    print(episode)


# # obtain the corresponding state-value function
# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
#          for k, v in Q.items())

# plot the state-value function
# plot_blackjack_values(V_to_plot)