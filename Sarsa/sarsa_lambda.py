#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 15:27:36
 * @Description: Basic implementation for Sarsa lambda
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
'''

from RL_base import RL
import numpy as np
import pandas as pd

class SarsaLambdaTable(RL):
    def __init__(self, actions, lr = 0.01, reward_decay = 0.9, e_greedy = 0.9, trace_decay = 0.9):
        super(SarsaLambdaTable, self).__init__(actions, lr, reward_decay, e_greedy)
        self._lambda = trace_decay
        self._eligbility_trace = self._q_table.copy()

    def learn(self, state, action, reward, state_, action_):
        self.state_exist(state_)
        q_predict = self._q_table.loc[state, action]
        if state_ != 'terminal':
            q_target = reward + self._gamma * self._q_table.loc[state_, action_]
        else:
            q_target = reward
        # update q_table
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # method 1
        # self._eligbility_trace.loc[state, action] += 1

        # method 2
        self._eligbility_trace.loc[state, :] *= 0
        self._eligbility_trace.loc[state, action] = 1

        # Q update
        self._q_table += self._lr * error * self._eligbility_trace

        # decay eligibility trace after update
        self._eligbility_trace *= self._gamma * self._lambda

    def state_exist(self, state):
        if state not in self._q_table.index:
            # append new state to q table
            append_items = pd.Series( 
                [0] * len(self._actions), 
                index = self._q_table.columns,
                name = state
                )
            self._q_table = self._q_table.append(append_items)
            self._eligbility_trace = self._eligbility_trace.append(append_items)