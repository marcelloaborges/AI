import gym
# import gym_pull
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
env = gym.make('SuperMarioBros-1-1-v0')
# env = gym.make('meta-SuperMarioBros-v0')

env.reset()
total_score = 0
while total_score < 32000:
    action = [0] * 6
    obs, reward, is_finished, info = env.step(action)
    env.render()
    total_score = info["total_reward"]