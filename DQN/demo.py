#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:34:42
 * @Description: Basic DQN pipeline for demo
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''

from maze_env import Maze
from dqn import DeepQNetwork

def run_maze():
    step = 0
    for episode in range(100):
        # initial observation
        observation = env.reset()
        
        while True:
            # fresh env
            env.render()

            # choose action based on observation
            action = RL.choose_action(observation)

            # take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # RL learn form this transition
            observation = observation_

            if done:
                break

            step += 1
    print("Game Over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features, lr=0.01, reward_decay=0.9,
                      e_greedy=0.9, replace_target_iter=200,memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()