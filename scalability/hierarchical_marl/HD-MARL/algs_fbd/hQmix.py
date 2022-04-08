# coding=utf-8
import threading
import random
import numpy as np
import tensorflow as tf
import datetime
import os
from collections import deque
import pickle

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=200,
            tau=0.01,
            memory_size=2000,
            batch_size=32,
            epsilon=1.0,
            e_greedy_increment=0.001,
            soft_replace=False,
            output_graph=False,
            GPU_divide=None,
            session=None,
            mark=None,
            is_dueling=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon
        self.tau = tau
        self.soft_replace = soft_replace
        self.is_dueling = is_dueling

        self.mark = '' if mark is None else mark
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_Q_Net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net' + self.mark)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net' + self.mark)

        with tf.variable_scope('replacement' + self.mark):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        with tf.variable_scope('soft_replacement' + self.mark):
            self.target_soft_replace_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in
                                           zip(t_params, e_params)]

        self.sess = session
        # if GPU_divide is None:
        #     self.sess = tf.Session()
        # else:
        #     config = tf.ConfigProto()
        #     config.gpu_options.per_process_gpu_memory_fraction = GPU_divide
        #     self.sess = tf.Session(config=config)

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("../logs/", self.sess.graph)

        # self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        # self.saver = tf.train.Saver(max_to_keep=10)

    def _build_Q_Net(self):
        # ------------------ all inputs ------------------------
        with tf.variable_scope('input_placeholders' + self.mark):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # FIXME 0822
        w_initializer, b_initializer = tf.random_normal_initializer(stddev=0.01), tf.constant_initializer(0.)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s,
                                  units=64,
                                  activation=tf.nn.relu,
                                  kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,
                                  name='fc1')

            if self.is_dueling:
                fc2_v = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='fc2_v')

                fc2_a = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='fc2_a')

                fc_v = tf.layers.dense(inputs=fc2_v,
                                       units=1,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='fc_v')

                fc_a = tf.layers.dense(inputs=fc2_a,
                                       units=self.n_actions,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='fc_a')

                self.Q = fc_v + (fc_a - tf.reduce_mean(fc_a, axis=1, keep_dims=True))
            else:
                fc2 = tf.layers.dense(inputs=fc1,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='fc2')

                self.Q = tf.layers.dense(inputs=fc2,
                                         units=self.n_actions,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='Q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s_,
                                  units=64,
                                  activation=tf.nn.relu,
                                  kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,
                                  name='fc1')

            if self.is_dueling:
                fc2_v = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='fc2_v')

                fc2_a = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='fc2_a')

                fc_v = tf.layers.dense(inputs=fc2_v,
                                       units=1,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='fc_v')

                fc_a = tf.layers.dense(inputs=fc2_a,
                                       units=self.n_actions,
                                       activation=tf.nn.relu,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='fc_a')

                self.Q_next = fc_v + (fc_a - tf.reduce_mean(fc_a, axis=1, keep_dims=True))
            else:
                fc2 = tf.layers.dense(inputs=fc1,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='fc2')

                self.Q_next = tf.layers.dense(inputs=fc2,
                                              units=self.n_actions,
                                              kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer,
                                              name='Q_next')

        with tf.variable_scope('q_target' + self.mark):
            q_target = self.r + self.gamma * tf.reduce_max(self.Q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval' + self.mark):
            self.a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1, name='a_indices')
            self.q_eval_wrt_a = tf.gather_nd(params=self.Q, indices=self.a_indices,
                                             name='q_eval_wrt_a')  # shape=(None, )
        with tf.variable_scope('loss' + self.mark):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            # tf.summary.scalar('loss' + self.mark, self.loss)
        with tf.variable_scope('train' + self.mark):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # TODO
        transition_list = s + [a, r] + s_
        transition = np.array(transition_list)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # TODO add success-greedy exploration for low-level actor
    def choose_action(self, observation, is_test=False, success_greedy=False, success_rate=0.0, beta=1.0):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if is_test:
            actions_value = self.sess.run(self.Q, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            if success_greedy:
                # FIXME linear combination of success_rate and epsilon as exploration
                expl = beta * (1 - success_rate) + (1 - beta) * self.epsilon
                if np.random.uniform() > expl:
                    # forward feed the observation and get q value for every actions
                    actions_value = self.sess.run(self.Q, feed_dict={self.s: observation})
                    action = np.argmax(actions_value)
                else:
                    action = np.random.randint(0, self.n_actions)
            else:
                if np.random.uniform() > self.epsilon:
                    # forward feed the observation and get q value for every actions
                    actions_value = self.sess.run(self.Q, feed_dict={self.s: observation})
                    action = np.argmax(actions_value)
                else:
                    action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # print(self.memory_counter)
        # check to replace target parameters
        if self.soft_replace:
            self.sess.run(self.target_soft_replace_op)
        elif self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            # FIXME if Memory not full, return
            # return
        batch_memory = self.memory[sample_index, :]

        f_s = np.array(batch_memory[:, :self.n_features]).reshape([-1, self.n_features])
        f_s_ = np.array(batch_memory[:, -self.n_features:]).reshape([-1, self.n_features])
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: f_s,
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: f_s_
            })

        # self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(self.epsilon)
        self.learn_step_counter += 1

        return cost


class metaQMIX:
    def __init__(
            self,
            n_actions,
            n_global_features,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=200,
            tau=0.01,
            memory_size=2000,
            batch_size=32,
            epsilon=1.0,
            e_greedy_increment=0.001,
            soft_replace=False,
            output_graph=False,
            GPU_divide=None,
            session=None,
            mark=None,
            is_dueling=False
    ):
        self.n_actions = n_actions
        self.n_global_features = n_global_features
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon
        self.tau = tau
        self.soft_replace = soft_replace
        self.is_dueling = is_dueling

        self.mark = '' if mark is None else mark
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s1, s2, s3, a1, a2, a3, r, s1_, s2_, s3_]
        self.memory = np.zeros((self.memory_size, self.n_features * 6 + 3 + 2 + self.n_global_features * 2))

        # consist of [target_net, evaluate_net]
        self._build_Q_Net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='meta_target_net' + self.mark)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='meta_eval_net' + self.mark)

        with tf.variable_scope('meta_replacement' + self.mark):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        with tf.variable_scope('meta_soft_replacement' + self.mark):
            self.target_soft_replace_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in
                                           zip(t_params, e_params)]

        self.sess = session
        self.cost_his = []

        # TODO
        # self.saver = tf.train.Saver(max_to_keep=10)

    def _build_Q_Net(self, is_dueling=False):
        # ------------------ all inputs ------------------------
        with tf.variable_scope('meta_input_placeholders' + self.mark):
            # FIXME 0821
            self.S = tf.placeholder(tf.float32, [None, self.n_global_features], name='meta_S')
            self.q1_m_ = tf.placeholder(tf.float32, [None, ], name='q1_value_next')
            self.q2_m_ = tf.placeholder(tf.float32, [None, ], name='q2_value_next')
            self.q3_m_ = tf.placeholder(tf.float32, [None, ], name='q3_value_next')
            # self.q_concat_ = tf.placeholder(tf.float32, [None, 3], name='q_value_concat_next')

            self.s1 = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s1')  # input State
            self.s2 = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s2')  # input State
            self.s3 = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s3')  # input State
            self.s1_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s1_')  # input Next State
            self.s2_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s2_')  # input Next State
            self.s3_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s3_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, ], name='meta_r')  # input Reward
            self.a1 = tf.placeholder(tf.int32, [None, ], name='meta_a1')  # input Action
            self.a2 = tf.placeholder(tf.int32, [None, ], name='meta_a2')  # input Action
            self.a3 = tf.placeholder(tf.int32, [None, ], name='meta_a3')  # input Action
            # TODO
            self.t = tf.placeholder(tf.float32, [None, ], name='meta_t')  # subtask duration

        # FIXME 0822
        w_initializer, b_initializer = tf.random_normal_initializer(stddev=0.01), tf.constant_initializer(0.)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('meta_eval_net' + self.mark):
            a1_fc1 = tf.layers.dense(inputs=self.s1,
                                     units=128,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a1_fc1')

            a2_fc1 = tf.layers.dense(inputs=self.s2,
                                     units=128,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a2_fc1')

            a3_fc1 = tf.layers.dense(inputs=self.s3,
                                     units=128,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a3_fc1')

            a1_fc2 = tf.layers.dense(inputs=a1_fc1,
                                     units=64,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a1_fc2')

            a2_fc2 = tf.layers.dense(inputs=a2_fc1,
                                     units=64,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a2_fc2')

            a3_fc2 = tf.layers.dense(inputs=a3_fc1,
                                     units=64,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='a3_fc2')

            self.Q1 = tf.layers.dense(inputs=a1_fc2,
                                      units=self.n_actions,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='Q1')

            self.Q2 = tf.layers.dense(inputs=a2_fc2,
                                      units=self.n_actions,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='Q2')

            self.Q3 = tf.layers.dense(inputs=a3_fc2,
                                      units=self.n_actions,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='Q3')

        # ------------------ build target_net ------------------
        with tf.variable_scope('meta_target_net' + self.mark):
            a1_fc1_ = tf.layers.dense(inputs=self.s1_,
                                      units=128,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a1_fc1_')

            a2_fc1_ = tf.layers.dense(inputs=self.s2_,
                                      units=128,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a2_fc1_')

            a3_fc1_ = tf.layers.dense(inputs=self.s3_,
                                      units=128,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a3_fc1_')

            a1_fc2_ = tf.layers.dense(inputs=a1_fc1_,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a1_fc2_')

            a2_fc2_ = tf.layers.dense(inputs=a2_fc1_,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a2_fc2_')

            a3_fc2_ = tf.layers.dense(inputs=a3_fc1_,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='a3_fc2_')

            self.Q1_ = tf.layers.dense(inputs=a1_fc2_,
                                       units=self.n_actions,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='Q1_')

            self.Q2_ = tf.layers.dense(inputs=a2_fc2_,
                                       units=self.n_actions,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='Q2_')

            self.Q3_ = tf.layers.dense(inputs=a3_fc2_,
                                       units=self.n_actions,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       name='Q3_')

        with tf.variable_scope('meta_mixed_network'):
            a1_indices = tf.stack([tf.range(tf.shape(self.a1)[0], dtype=tf.int32), self.a1], axis=1, name='a1_indices')
            a2_indices = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), self.a2], axis=1, name='a2_indices')
            a3_indices = tf.stack([tf.range(tf.shape(self.a3)[0], dtype=tf.int32), self.a3], axis=1, name='a3_indices')
            q1_a = tf.gather_nd(params=self.Q1, indices=a1_indices, name='q1_eval_wrt_a')  # shape=(None, )
            q2_a = tf.gather_nd(params=self.Q2, indices=a2_indices, name='q2_eval_wrt_a')  # shape=(None, )
            q3_a = tf.gather_nd(params=self.Q3, indices=a3_indices, name='q3_eval_wrt_a')  # shape=(None, )
            self.q_concat = tf.stack([q1_a, q2_a, q3_a], axis=1, name='q_concat')
            self.q_concat_ = tf.stack([self.q1_m_, self.q2_m_, self.q3_m_], axis=1, name='q_concat_next')

            # FIXME 0821 mixed network
            # FIXME layer 1
            non_abs_w1 = tf.layers.dense(inputs=self.S,
                                         units=3 * 32,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='non_abs_w1')
            self.w1 = tf.reshape(tf.abs(non_abs_w1), shape=[-1, 3, 32], name='w1')
            self.b1 = tf.layers.dense(inputs=self.S,
                                      units=32,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='non_abs_b1')

            # FIXME layer 2
            non_abs_w2 = tf.layers.dense(inputs=self.S,
                                         units=32 * 1,
                                         kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer,
                                         name='non_abs_w2')
            self.w2 = tf.reshape(tf.abs(non_abs_w2), shape=[-1, 32, 1], name='w2')
            bef_b2 = tf.layers.dense(inputs=self.S,
                                     units=32,
                                     activation=tf.nn.relu,
                                     kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer,
                                     name='bef_b2')
            self.b2 = tf.layers.dense(inputs=bef_b2,
                                      units=1,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer,
                                      name='non_abs_b2')

            # FIXME 0825 f(Q1 + Q2 + Q3)
            # FIXME 0821 [n, 1, 3] x [n, 3, 32] -> [n, 1, 32]
            lin1 = tf.matmul(tf.reshape(self.q_concat, shape=[-1, 1, 3]), self.w1) \
                   + tf.reshape(self.b1, shape=[-1, 1, 32])
            a1 = tf.nn.elu(lin1, name='a1')

            # FIXME 0821 [n, 1, 32] x [n, 32, 1] -> [n, 1, 1] + [n, 1]
            self.Q_tot = tf.reshape(tf.matmul(a1, self.w2), shape=[-1, 1]) + self.b2

            # FIXME 0825 f(Q1_ + Q2_ + Q3_)
            # FIXME 0821 [n, 1, 3] x [n, 3, 32] -> [n, 1, 32]
            lin1_ = tf.matmul(tf.reshape(self.q_concat_, shape=[-1, 1, 3]), self.w1) \
                   + tf.reshape(self.b1, shape=[-1, 1, 32])
            a1_ = tf.nn.elu(lin1_, name='a1')

            # FIXME 0821 [n, 1, 32] x [n, 32, 1] -> [n, 1, 1] + [n, 1]
            self.Q_tot_ = tf.reshape(tf.matmul(a1_, self.w2), shape=[-1, 1]) + self.b2
            # self.Q_tot_ = tf.stop_gradient(Q_tot_)

        with tf.variable_scope('meta_q_target' + self.mark):
            # self.Q_tot_ = tf.placeholder(tf.float32, [None, ], name='Q_tot_')
            q_target = self.r + tf.pow(x=self.gamma, y=self.t) * self.Q_tot_
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('meta_loss' + self.mark):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.Q_tot, name='TD_error'))
            # tf.summary.scalar('loss' + self.mark, self.loss)
        with tf.variable_scope('meta_train' + self.mark):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s1, s2, s3, a1, a2, a3, r, t, s1_, s2_, s3_, S, S_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # FIXME 0821
        transition_list = s1 + s2 + s3 + s1_ + s2_ + s3_ + [a1, a2, a3, r, t] + S + S_
        transition = np.array(transition_list).reshape((-1,))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, index, is_test=False):
        if index not in [0, 1, 2]:
            print('--FATAL ERROR: Wrong index when choosing action.')
            return None

        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if is_test:
            if index == 0:
                actions_value = self.sess.run(self.Q1, feed_dict={self.s1: observation})
            elif index == 1:
                actions_value = self.sess.run(self.Q2, feed_dict={self.s2: observation})
            else:
                actions_value = self.sess.run(self.Q3, feed_dict={self.s3: observation})
            action = np.argmax(actions_value)
        else:
            if np.random.uniform() > self.epsilon:
                # forward feed the observation and get q value for every actions
                if index == 0:
                    actions_value = self.sess.run(self.Q1, feed_dict={self.s1: observation})
                elif index == 1:
                    actions_value = self.sess.run(self.Q2, feed_dict={self.s2: observation})
                else:
                    actions_value = self.sess.run(self.Q3, feed_dict={self.s3: observation})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # check to replace target parameters
        if self.soft_replace:
            self.sess.run(self.target_soft_replace_op)
        elif self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        # sample batch memory from all memory
        if self.memory_counter < self.memory_size:
            print('INFO: Memory not full yet.')
            return

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # FIXME further check
        f_s1 = np.array(batch_memory[:, :self.n_features]).reshape([-1, self.n_features])
        f_s2 = np.array(batch_memory[:, self.n_features:self.n_features * 2]).reshape([-1, self.n_features])
        f_s3 = np.array(batch_memory[:, self.n_features * 2:self.n_features * 3]).reshape([-1, self.n_features])
        f_s1_ = np.array(batch_memory[:, self.n_features * 3:self.n_features * 4]).reshape([-1, self.n_features])
        f_s2_ = np.array(batch_memory[:, self.n_features * 4:self.n_features * 5]).reshape([-1, self.n_features])
        f_s3_ = np.array(batch_memory[:, self.n_features * 5:self.n_features * 6]).reshape([-1, self.n_features])

        a1 = batch_memory[:, self.n_features * 6].astype(int)
        a2 = batch_memory[:, self.n_features * 6 + 1].astype(int)
        a3 = batch_memory[:, self.n_features * 6 + 2].astype(int)
        r = batch_memory[:, self.n_features * 6 + 3]
        t = batch_memory[:, self.n_features * 6 + 4]

        S = np.array(batch_memory[:, -self.n_global_features * 2:-self.n_global_features]).reshape([-1, self.n_global_features])
        S_ = np.array(batch_memory[:, -self.n_global_features:]).reshape([-1, self.n_global_features])

        # FIXME 0825
        q1_, q2_, q3_ = self.sess.run([self.Q1_, self.Q2_, self.Q3_], feed_dict={self.s1_: f_s1_,
                                                                                 self.s2_: f_s2_,
                                                                                 self.s3_: f_s3_})
        q1_m_ = np.max(q1_, axis=1)
        q2_m_ = np.max(q2_, axis=1)
        q3_m_ = np.max(q3_, axis=1)

        q_tot_ = self.sess.run(self.Q_tot_, feed_dict={self.S: S_,
                                                       self.q1_m_: q1_m_,
                                                       self.q2_m_: q2_m_,
                                                       self.q3_m_: q3_m_})

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.S: S,
                self.s1: f_s1,
                self.s2: f_s2,
                self.s3: f_s3,
                self.a1: a1,
                self.a2: a2,
                self.a3: a3,
                self.r: r,
                self.t: t,
                self.Q_tot_: q_tot_
            })

        # self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(self.epsilon)
        self.learn_step_counter += 1

        return cost


