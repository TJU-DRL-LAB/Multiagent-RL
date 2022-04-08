import numpy as np
import tensorflow as tf

# SEED = 222
# np.random.seed(sd.SEED)
# tf.set_random_seed(sd.SEED)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=300,
            tau=0.01,
            memory_size=2000,
            batch_size=32,
            epsilon=1.0,
            e_greedy_increment=0.001,
            soft_replace=False,
            output_graph=False,
            GPU_divide=None,
            session=None,
            mark=None
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
            self.target_soft_replace_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(t_params, e_params)]

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
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            fc2 = tf.layers.dense(inputs=fc1,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                  name='fc2')

            self.Q = tf.layers.dense(inputs=fc2,
                                     units=self.n_actions,
                                     kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                     name="Q")


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s_,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            fc2 = tf.layers.dense(inputs=fc1,
                              units=128,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                              name='fc2')

            self.Q_next = tf.layers.dense(inputs=fc2,
                                units=self.n_actions,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                name="Q_next")


        with tf.variable_scope('q_target' + self.mark):
            q_target = self.r + self.gamma * tf.reduce_max(self.Q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval' + self.mark):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1, name='a_indices')
            self.q_eval_wrt_a = tf.gather_nd(params=self.Q, indices=a_indices, name='q_eval_wrt_a')    # shape=(None, )
        with tf.variable_scope('loss' + self.mark):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            # tf.summary.scalar('loss' + self.mark, self.loss)
        with tf.variable_scope('train' + self.mark):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, a, r, s_,):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # TODO
        transition_list = s + [a, r] + s_
        transition = np.array(transition_list)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.Q, feed_dict={self.s: observation})
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
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
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

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(self.epsilon)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



class metaDeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00025,
            reward_decay=0.99,
            e_greedy=0.1,
            replace_target_iter=300,
            tau=0.01,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=0.001,
            soft_replace=False,
            output_graph=False,
            GPU_divide=None,
            session=None,
            mark=None
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
        self.epsilon = 1.0
        self.tau = tau
        self.soft_replace = soft_replace

        self.mark = '' if mark is None else mark
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))

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

    def _build_Q_Net(self):
        # ------------------ all inputs ------------------------
        with tf.variable_scope('meta_input_placeholders' + self.mark):
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s')  # input State
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='meta_s_')  # input Next State
            self.r = tf.placeholder(tf.float32, [None, 1], name='meta_r')  # input Reward
            self.a = tf.placeholder(tf.int32, [None, ], name='meta_a')  # input Action
            self.t = tf.placeholder(tf.float32, [None, ], name='meta_t')  # subtask duration

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('meta_eval_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            fc2 = tf.layers.dense(inputs=fc1,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                  name='fc2')

            self.Q = tf.layers.dense(inputs=fc2,
                                     units=self.n_actions,
                                     kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                     name='Q')


        # ------------------ build target_net ------------------
        with tf.variable_scope('meta_target_net' + self.mark):
            fc1 = tf.layers.dense(inputs=self.s_,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.n_features)),
                                  name='fc1')

            fc2 = tf.layers.dense(inputs=fc1,
                                  units=128,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                  name='fc2')

            self.Q_next = tf.layers.dense(inputs=fc2,
                                          units=self.n_actions,
                                          kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 128)),
                                          name='Q_next')


        with tf.variable_scope('meta_q_target' + self.mark):
            # TODO
            q_target = self.r + tf.pow(x=self.gamma, y=self.t) * tf.reduce_max(self.Q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('meta_q_eval' + self.mark):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1, name='a_indices')
            self.q_eval_wrt_a = tf.gather_nd(params=self.Q, indices=a_indices, name='q_eval_wrt_a')    # shape=(None, )
        with tf.variable_scope('meta_loss' + self.mark):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
            # tf.summary.scalar('loss' + self.mark, self.loss)
        with tf.variable_scope('meta_train' + self.mark):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, a, r, t, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # TODO
        transition_list = s + [a, r, t] + s_
        transition = np.array(transition_list)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.Q, feed_dict={self.s: observation})
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
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        f_s = np.array(batch_memory[:, :self.n_features]).reshape([-1, self.n_features])
        f_s_ = np.array(batch_memory[:, -self.n_features:]).reshape([-1, self.n_features])
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: f_s,
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.t: batch_memory[:, self.n_features + 2],
                self.s_: f_s_
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon - self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(self.epsilon)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

# sess = tf.Session()

# game_size = 9
# n_features = game_size*game_size + 2
# goal_num = 5
# nav_action_num = 5
# meta_lr, lr = 0.00025, 0.0005
# meta_gamma, gamma = 0.99, 0.95
# meta_mem, mem = 10000, 10000
# meta_batch, batch = 32, 32
# meta_dqn1 = metaDeepQNetwork(n_features=n_features,
#                     n_actions=goal_num,
#                     learning_rate=meta_lr,
#                     reward_decay=meta_gamma,
#                     replace_target_iter=100,
#                     memory_size=meta_mem,
#                     batch_size=meta_batch,
#                     e_greedy_increment=0.00001,
#                     session=sess,
#                     mark='1',
#                     # output_graph=True
#                     )
#
# dqn1 = DeepQNetwork(n_features=n_features,
#            n_actions=nav_action_num,
#            learning_rate=lr,
#            reward_decay=gamma,
#            replace_target_iter=100,
#            memory_size=mem,
#            batch_size=batch,
#            e_greedy_increment=0.0001,
#            session=sess,
#            mark='1',
#            # output_graph=True
#            )
#
# meta_dqn2 = metaDeepQNetwork(n_features=n_features,
#                              n_actions=goal_num,
#                              learning_rate=meta_lr,
#                              reward_decay=meta_gamma,
#                              replace_target_iter=100,
#                              memory_size=meta_mem,
#                              batch_size=meta_batch,
#                              e_greedy_increment=0.00001,
#                              session=sess,
#                              mark='2',
#                              # output_graph=True
#                              )
#
# dqn2 = DeepQNetwork(n_features=n_features,
#                     n_actions=nav_action_num,
#                     learning_rate=lr,
#                     reward_decay=gamma,
#                     replace_target_iter=100,
#                     memory_size=mem,
#                     batch_size=batch,
#                     e_greedy_increment=0.0001,
#                     session=sess,
#                     mark='2',
#                     output_graph=True
#                     )

