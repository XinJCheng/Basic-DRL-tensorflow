'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-22 19:39:28
 * @LastEditors: Xinjing Cheng
 * @LastEditTime: 2018-07-22 19:40:03
 * @Description: Native Q table Learning 
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
         self.epsilon = e_greedy
         self.q_table = pd.DataFram(columns = self._actions, dtype = np.float64)
         
    def action(self, observation):
        # check if the observation has exist in q_table
        self.state_exist(observation)

    def learn(self, state, action, reward, state_)
    def state_exist(self, state)
