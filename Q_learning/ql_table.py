#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:35:33
 * @Description: Basic implementation for Q table
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''


import numpy as np
import pandas as pd 

class QLearningTable:
    def __init__(self, actions, lr = 0.01, reward_decay= 0.9, e_greedy = 0.9):
         self._actions = actions
         self._lr = lr
         self._gamma = reward_decay
         self._epsilon = e_greedy
         self._q_table = pd.DataFrame(columns = self._actions, dtype = np.float64)

    # return action index given observation     
    def action(self, observation):
        # check if the observation has exist in q_table
        self.state_exist(observation)
        # action select
        if np.random.uniform() < self._epsilon:
            # choose best action with max reward
            state_action = self._q_table.loc[observation, :]
            # solve the problem where actions have some reward and always choose
            # the same action 
            state_action = state_action.reindex(
                np.random.permutation(state_action.index)
            )
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self._actions)
        return action

    # update q_table
    def learn(self, state, action, reward, state_):
        self.state_exist(state_)
        q_predict = self._q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = reward + self._gamma * self._q_table.loc[state_, :].max()
        else:
            q_target = reward
        # update q_table
        self._q_table.loc[state, action] += self._lr * (q_target - q_predict)

    def state_exist(self, state):
        if state not in self._q_table.index:
            # append new state to q table
            self._q_table = self._q_table.append(
                pd.Series(
                    [0] * len(self._actions), 
                    index = self._q_table.columns,
                    name = state
                )
            )