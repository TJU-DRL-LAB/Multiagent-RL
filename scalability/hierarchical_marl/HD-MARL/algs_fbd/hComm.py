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

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s,
                                  units=64,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            if self.is_dueling:
                fc2_v = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                        name='fc2_v')

                fc2_a = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                        name='fc2_a')

                fc_v = tf.layers.dense(inputs=fc2_v,
                                       units=1,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                       name='fc_v')

                fc_a = tf.layers.dense(inputs=fc2_a,
                                       units=self.n_actions,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                       name='fc_a')

                self.Q = fc_v + (fc_a - tf.reduce_mean(fc_a, axis=1, keep_dims=True))
            else:
                fc2 = tf.layers.dense(inputs=fc1,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                      name='fc2')

                self.Q = tf.layers.dense(inputs=fc2,
                                         units=self.n_actions,
                                         kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                         name='Q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s_,
                                  units=64,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            if self.is_dueling:
                fc2_v = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                        name='fc2_v')

                fc2_a = tf.layers.dense(inputs=fc1,
                                        units=32,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                        name='fc2_a')

                fc_v = tf.layers.dense(inputs=fc2_v,
                                       units=1,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                       name='fc_v')

                fc_a = tf.layers.dense(inputs=fc2_a,
                                       units=self.n_actions,
                                       activation=tf.nn.relu,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 32)),
                                       name='fc_a')

                self.Q_next = fc_v + (fc_a - tf.reduce_mean(fc_a, axis=1, keep_dims=True))
            else:
                fc2 = tf.layers.dense(inputs=fc1,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                      name='fc2')

                self.Q_next = tf.layers.dense(inputs=fc2,
                                              units=self.n_actions,
                                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
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


class metaCOMDQN:
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

        # initialize zero memory [s1, s2, s3, a1, a2, a3, r, s1_, s2_, s3_]
        self.memory1 = np.zeros((self.memory_size, self.n_features * 4 + 1 + 2))
        self.memory2 = np.zeros((self.memory_size, self.n_features * 4 + 1 + 2))
        self.memory_counter1 = 0
        self.memory_counter2 = 0

        # FIXME 1108
        # self.mem_locks = [threading.Lock() for i in range(3)]
        # self.net_locks = [threading.Lock() for i in range(3)]

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
            self.s1 = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s1')  # input State
            self.s2 = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s2')  # input State
            self.s1_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s1_')  # input Next State
            self.s2_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s2_')  # input Next State
            self.r1 = tf.placeholder(tf.float32, [None, ], name='meta_r1')  # input Reward
            self.r2 = tf.placeholder(tf.float32, [None, ], name='meta_r2')  # input Reward
            self.a1 = tf.placeholder(tf.int32, [None, ], name='meta_a1')  # input Action
            self.a2 = tf.placeholder(tf.int32, [None, ], name='meta_a2')  # input Action
            # TODO
            self.t1 = tf.placeholder(tf.float32, [None, ], name='meta_t1')  # subtask duration
            self.t2 = tf.placeholder(tf.float32, [None, ], name='meta_t2')  # subtask duration

            # FIXME 0822
            self.c1 = tf.placeholder(tf.float32, [None, 64], name='meta_c1')
            self.c2 = tf.placeholder(tf.float32, [None, 64], name='meta_c2')
            self.c1_ = tf.placeholder(tf.float32, [None, 64], name='meta_c1_')
            self.c2_ = tf.placeholder(tf.float32, [None, 64], name='meta_c2_')

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('meta_eval_net' + self.mark):
            self.a1_fc1 = tf.layers.dense(inputs=self.s1,
                                          units=64,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                          name='a1_fc1')

            self.a2_fc1 = tf.layers.dense(inputs=self.s2,
                                          units=64,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                          name='a2_fc1')

            # FIXME 0822
            c1 = tf.concat([self.a1_fc1, self.c1], axis=1, name='concat_input1')
            c2 = tf.concat([self.a2_fc1, self.c2], axis=1, name='concat_input2')

            # FIXME 0824
            c1 = tf.stop_gradient(c1)
            c2 = tf.stop_gradient(c2)

            a1_fc2 = tf.layers.dense(inputs=c1,
                                     units=64,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                     name='a1_fc2')

            a2_fc2 = tf.layers.dense(inputs=c2,
                                     units=64,
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                     name='a2_fc2')

            self.Q1 = tf.layers.dense(inputs=a1_fc2,
                                      units=self.n_actions,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                      name='Q1')

            self.Q2 = tf.layers.dense(inputs=a2_fc2,
                                      units=self.n_actions,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                      name='Q2')

        # ------------------ build target_net ------------------
        with tf.variable_scope('meta_target_net' + self.mark):
            self.a1_fc1_ = tf.layers.dense(inputs=self.s1_,
                                           units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                           name='a1_fc1_')

            self.a2_fc1_ = tf.layers.dense(inputs=self.s2_,
                                           units=64,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                           name='a2_fc1_')

            # FIXME 0822
            c1_ = tf.concat([self.a1_fc1_, self.c1_], axis=1, name='concat_input1_')
            c2_ = tf.concat([self.a2_fc1_, self.c2_], axis=1, name='concat_input2_')

            # FIXME 0824
            c1_ = tf.stop_gradient(c1_)
            c2_ = tf.stop_gradient(c2_)

            a1_fc2_ = tf.layers.dense(inputs=c1_,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                      name='a1_fc2_')

            a2_fc2_ = tf.layers.dense(inputs=c2_,
                                      units=64,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                      name='a2_fc2_')

            self.Q1_ = tf.layers.dense(inputs=a1_fc2_,
                                       units=self.n_actions,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                       name='Q1_')

            self.Q2_ = tf.layers.dense(inputs=a2_fc2_,
                                       units=self.n_actions,
                                       kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)),
                                       name='Q2_')


        with tf.variable_scope('meta_q_target' + self.mark):
            # TODO
            q_target1 = self.r1 + tf.pow(x=self.gamma, y=self.t1) * (tf.reduce_max(self.Q1_, axis=1, name='Q1_max_s_'))
            q_target2 = self.r2 + tf.pow(x=self.gamma, y=self.t2) * (tf.reduce_max(self.Q2_, axis=1, name='Q2_max_s_'))

            self.q_target1 = tf.stop_gradient(q_target1)
            self.q_target2 = tf.stop_gradient(q_target2)

        with tf.variable_scope('meta_q_eval' + self.mark):
            a1_indices = tf.stack([tf.range(tf.shape(self.a1)[0], dtype=tf.int32), self.a1], axis=1, name='a1_indices')
            a2_indices = tf.stack([tf.range(tf.shape(self.a2)[0], dtype=tf.int32), self.a2], axis=1, name='a2_indices')
            self.q1_eval_wrt_a = tf.gather_nd(params=self.Q1, indices=a1_indices,
                                              name='q1_eval_wrt_a')  # shape=(None, )
            self.q2_eval_wrt_a = tf.gather_nd(params=self.Q2, indices=a2_indices,
                                              name='q2_eval_wrt_a')  # shape=(None, )

        with tf.variable_scope('meta_loss' + self.mark):
            self.loss1 = tf.reduce_mean(tf.squared_difference(self.q_target1, self.q1_eval_wrt_a, name='TD_error1'))
            self.loss2 = tf.reduce_mean(tf.squared_difference(self.q_target2, self.q2_eval_wrt_a, name='TD_error2'))

        with tf.variable_scope('meta_train' + self.mark):
            self._train_op1 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss1)
            self._train_op2 = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss2)

    # FIXME 0822
    def store_transition(self, s1, s2, a, r, t, s1_, s2_, id):
        transition_list = s1 + s2 + s1_ + s2_ + [a, r, t]
        transition = np.array(transition_list).reshape((-1,))

        if id == 0:
            # FIXME 1108
            # self.mem_locks[id].acquire()
            index = self.memory_counter1 % self.memory_size
            self.memory1[index, :] = transition
            self.memory_counter1 += 1
            # self.mem_locks[id].release()
        elif id == 1:
            # self.mem_locks[id].acquire()
            index = self.memory_counter2 % self.memory_size
            self.memory2[index, :] = transition
            self.memory_counter2 += 1
            # self.mem_locks[id].release()
        else:
            print('ERROR: Wrong agent id!')

        # print(self.memory_counter1)
        # print(self.memory_counter2)
        # print(self.memory_counter3)
        # print('---')

    def choose_action(self, observation, index, is_test=False):
        if index not in [0, 1]:
            print('--FATAL ERROR: Wrong index when choosing action.')
            return None
        # to have batch dimension when feed into tf placeholder

        # FIXME 1108
        # self.net_locks[index].acquire()
        s1 = np.array(observation[0]).reshape((1, -1))
        s2 = np.array(observation[1]).reshape((1, -1))

        # FIXME 0822 high-level communication
        a1_fc1, a2_fc1 = self.sess.run([self.a1_fc1, self.a2_fc1],
                                       feed_dict={self.s1: s1, self.s2: s2})

        c1 = a2_fc1
        c2 = a1_fc1

        if is_test:
            if index == 0:
                actions_value = self.sess.run(self.Q1, feed_dict={self.s1: s1, self.c1: c1})
            else:
                actions_value = self.sess.run(self.Q2, feed_dict={self.s2: s2, self.c2: c2})
            action = np.argmax(actions_value)
        else:
            if np.random.uniform() > self.epsilon:
                # forward feed the observation and get q value for every actions

                if index == 0:
                    actions_value = self.sess.run(self.Q1, feed_dict={self.s1: s1, self.c1: c1})
                else:
                    actions_value = self.sess.run(self.Q2, feed_dict={self.s2: s2, self.c2: c2})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)

        # self.net_locks[index].release()
        return action

    def learn(self):
        # check to replace target parameters
        if self.soft_replace:
            self.sess.run(self.target_soft_replace_op)
        elif self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)

        if self.memory_counter1 < self.memory_size \
                or self.memory_counter2 < self.memory_size:
            print('INFO: Memory not full yet.')
            return

        cost = []

        # FIXME 0828 non-consistent experience
        # FIXME 1106
        sample_index1 = np.random.choice(min(self.memory_counter1, self.memory_size), size=self.batch_size)
        sample_index2 = np.random.choice(min(self.memory_counter2, self.memory_size), size=self.batch_size)

        batch_memory1 = self.memory1[sample_index1, :]
        batch_memory2 = self.memory2[sample_index2, :]

        # FIXME train a1
        f1_s1 = np.array(batch_memory1[:, :self.n_features]).reshape([-1, self.n_features])
        f1_s2 = np.array(batch_memory1[:, self.n_features:self.n_features * 2]).reshape([-1, self.n_features])
        f1_s1_ = np.array(batch_memory1[:, self.n_features * 2:self.n_features * 3]).reshape([-1, self.n_features])
        f1_s2_ = np.array(batch_memory1[:, self.n_features * 3:self.n_features * 4]).reshape([-1, self.n_features])

        f1_a = batch_memory1[:, -3]
        f1_r = batch_memory1[:, -2]
        f1_t = batch_memory1[:, -1]

        # FIXME 0827 high-level communication
        f1_a2_fc1 = self.sess.run(self.a2_fc1, feed_dict={self.s2: f1_s2})
        c1 = f1_a2_fc1
        f1_a2_fc1_ = self.sess.run(self.a2_fc1_, feed_dict={self.s2_: f1_s2_})
        c1_ = f1_a2_fc1_

        _, cost1 = self.sess.run(
            [self._train_op1, self.loss1],
            feed_dict={
                self.s1: f1_s1,
                self.c1: c1,
                self.a1: f1_a,
                self.r1: f1_r,
                self.t1: f1_t,
                self.s1_: f1_s1_,
                self.c1_: c1_,
            })
        cost.append(cost1)

        # FIXME train a2
        f2_s1 = np.array(batch_memory2[:, :self.n_features]).reshape([-1, self.n_features])
        f2_s2 = np.array(batch_memory2[:, self.n_features:self.n_features * 2]).reshape([-1, self.n_features])
        f2_s1_ = np.array(batch_memory2[:, self.n_features * 2:self.n_features * 3]).reshape([-1, self.n_features])
        f2_s2_ = np.array(batch_memory2[:, self.n_features * 3:self.n_features * 4]).reshape([-1, self.n_features])

        f2_a = batch_memory2[:, -3]
        f2_r = batch_memory2[:, -2]
        f2_t = batch_memory2[:, -1]

        # FIXME 0827 high-level communication
        f2_a1_fc1 = self.sess.run(self.a1_fc1, feed_dict={self.s1: f2_s1})
        c2 = f2_a1_fc1
        f2_a1_fc1_ = self.sess.run(self.a1_fc1_, feed_dict={self.s1_: f2_s1_})
        c2_ = f2_a1_fc1_

        _, cost2 = self.sess.run(
            [self._train_op2, self.loss2],
            feed_dict={
                self.s2: f2_s2,
                self.c2: c2,
                self.a2: f2_a,
                self.r2: f2_r,
                self.t2: f2_t,
                self.s2_: f2_s2_,
                self.c2_: c2_,
            })
        cost.append(cost2)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(self.epsilon)
        self.learn_step_counter += 1

        return cost



