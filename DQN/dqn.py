'''
 * @Author: Xinjing Cheng
 * @Date: 2018-07-23 17:17:29
 * @Description: Baisc implementation of DQN
 * @Email: cnorbot@gmail.com
 * @Company: Baidu Research, Baidu Inc.
'''
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 

class DeepQNetwork:
    def __init__(self, n_actions, n_features, lr = 0.01, reward_decay = 0.9, 
                 e_greedy = 0.9, replace_target_iter = 300, memory_size = 500, 
                 batch_size = 32,e_greedy_increment = None, output_graph = False):
        self._n_actions = n_actions
        self._n_features = n_features
        self._lr = lr 
        self._gamma = reward_decay 
        self._epsilon_max = e_greedy 
        self._replace_target_iter = replace_target_iter 
        self._memory_size = memory_size
        self._batch_size = batch_size
        self._epsilon_increment = e_greedy_increment 
        self._epsilon = 0 if e_greedy_increment is not None else self._epsilon_max

        # total learning step
        self._learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self._memory = np.zeros((self._memory_size, n_features * 2 + 2))

        # build target_net and eval_net
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        self.sees = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sees.graph)

        self.sees.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # all input variables
        self.s = tf.placeholder(tf.float32, [None, self._n_features], name = 's')
        self.s_ = tf.placeholder(tf.float32, [None, self._n_features], name = 's_')
        self.r = tf.placeholder(tf.float32, [None, ], name = 'r')
        self.a = tf.placeholder(tf.int32, [None, ], name = 'a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        
        # build eval net
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 20, tf.nn.relu,
            kernel_initializer = w_initializer, bias_initializer = b_initializer, 
            name = 'e1')
            self.q_eval = tf.layers.dense(e1, self._n_actions, 
            kernel_initializer = w_initializer, bias_initializer = b_initializer,
            name = 'q')

        # build target net
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 20, tf.nn.relu,
            kernel_initializer = w_initializer, bias_initializer = b_initializer, 
            name = 't1')
            self.q_next = tf.layers.dense(t1, self._n_actions, 
            kernel_initializer = w_initializer, bias_initializer = b_initializer,
            name = 't2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self._gamma * tf.reduce_max(self.q_next, axis = 1, name = 'Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype = tf.int32), self.a], axis = 1)
            self.q_eval_wrt_a = tf.gather_nd(params = self.q_eval, indices = a_indices)
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a), name = 'TD_error')

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self._lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self._memory_size
        self._memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # get the batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self._epsilon_max:
            # forward feed the observation and get q value for every actions
            actions_value = self.sees.run(self.q_eval,
                                          feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self._n_actions)
        return action
    
    def learn(self):
        # check to replace target parameters
        if self._learn_step_counter % self._replace_target_iter == 0:
            self.sees.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
        
        # simple batch memory from all memory
        if self.memory_counter > self._memory_size:
            sample_index = np.random.choice(self._memory_size, size = self._batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size = self._batch_size)
        batch_memory = self._memory[sample_index, :]

        _, cost = self.sees.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self._n_features],
                self.a: batch_memory[:, self._n_features],
                self.r: batch_memory[:, self._n_features + 1],
                self.s_: batch_memory[:, -self._n_features:],
            }
        )
        self.cost_his.append(cost)

        # increse epsilion
        self._epsilon = self._epsilon + self._epsilon_increment \
        if self._epsilon < self._epsilon_max \
        else self._epsilon_max

        self._learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph = True)

        

        



