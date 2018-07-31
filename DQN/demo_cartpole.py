#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-31 11:34:42
 * @Description: Basic DQN pipeline for OpenAI gym CartPole-v0
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''

import gym
from dqn import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    lr = 0.01,
    e_greedy = 0.9,
    replace_target_iter = 100,
    memory_size = 2000,
    e_greedy_increment = 0.001
    )

total_steps = 0

for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # change to a more reasonable reward function
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()
        
        if done:
            print(
                'episode: ', i_episode,
                'ep_r: ', round(ep_r, 2),
                'epsilon: ', round(RL._epsilon, 2)
            )
            break
        
        observation = observation_
        total_steps += 1

RL.plot_cost()
