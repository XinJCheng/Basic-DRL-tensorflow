#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:35:33
 * @Description: Basic implementation of base class for RL
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''
 

import numpy as np
import pandas as pd 

class RL(object):
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

    def learn(self, *args):
        pass

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
    def show_table(self):
        print (self._q_table)