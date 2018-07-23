#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 11:35:33
 * @Description: Basic implementation for Sarsa
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
 '''
 

import numpy as np
import pandas as pd
from RL_base import RL 

class SarsaTable(RL):
    def __init__(self, actions, lr = 0.01, reward_decay= 0.9, e_greedy = 0.9):
        super(SarsaTable, self).__init__(actions, lr, reward_decay, e_greedy)

    # update q_table
    def learn(self, state, action, reward, state_, action_):
        self.state_exist(state_)
        q_predict = self._q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = reward + self._gamma * self._q_table.loc[state_, action_]
        else:
            q_target = reward
        # update q_table
        self._q_table.loc[state, action] += self._lr * (q_target - q_predict)