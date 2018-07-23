#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:34:42
 * @Description: Basic Sarsa pipeline for demo
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''

from maze_env import Maze
from sarsa import SarsaTable
from sarsa_lambda import SarsaLambdaTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # choose action based on observation
        action = RL.action(str(observation))
        
        while True:
            # fresh env
            env.render()

            # take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            # choose action based on observation
            action_ = RL.action(str(observation_))

            # RL learn form this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_

            # show Q table 
            RL.show_table()
            if done:
                break

    print ("Game Over")
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # RL = SarsaTable(actions = list(range(env.n_actions)))
    RL = SarsaLambdaTable(actions = list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()        