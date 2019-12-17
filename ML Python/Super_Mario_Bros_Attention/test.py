import gym

env = gym.make('procgen:procgen-coinrun-v0')
obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        break