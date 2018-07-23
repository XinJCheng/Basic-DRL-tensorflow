#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:34:42
 * @Description: Basic Q learning pipeline for demo
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''

from maze_env import Maze
from ql_table import QLearningTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        
        while True:
            # fresh env
            env.render()

            # choose action based on observation
            action = RL.action(str(observation))

            # take action and get next observation and reward
            observation_, reward, done = env.step(action)
            

            # RL learn form this transition
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            # show Q table 
            RL.show_table()
            if done:
                break

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()        