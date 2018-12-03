import gym
env = gym.make('PongDeterministic-v4')
print('observation space:', env.observation_space)
print('action space:', env.action_space)


state = env.reset()

print(state)
print('end')

# for _ in range(1):
#     while True:
#         env.render()
#         action = env.action_space.sample()
#         next_state, reward, done, info = env.step(action)
#         print(state, action, next_state, reward)
#         next_state = state        
#         if done:
#             break        