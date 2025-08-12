# simple dqn style agent skeleton using gym
import gym
import random
env = gym.make('CartPole-v1')
s = env.reset()
for _ in range(5):
    a = env.action_space.sample()
    s, r, done, info = env.step(a)
    if done:
        s = env.reset()
print('sample run done')
