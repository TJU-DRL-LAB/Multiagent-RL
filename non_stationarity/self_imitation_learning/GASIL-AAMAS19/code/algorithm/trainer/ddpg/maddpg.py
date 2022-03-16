import numpy as np
import tensorflow as tf

import algorithm.common.tf_utils as U
from algorithm.trainer import AgentTrainer
from algorithm.common.distributions2 import make_pdtype
from algorithm.common.reinforce_utils import make_update_exp
from algorithm.prioritized_experience_replay_buffer.utils import add_episode

FLAGS = tf.app.flags.FLAGS  # alias

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        weight_ph = tf.placeholder(tf.float32, [None], name="important_weight")
        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func",
                   num_units=FLAGS.num_units, layer_norm=layer_norm)

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.lambda2), p_func_vars)

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        # TODO: 这里添加了 deterministic action
        determin_act_sample, act_sample = act_pd.sample(deterministic=True)
        # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # act_pd.mode() sample action from current policy

        # build q-function input
        # print("no adv state info, no adv action info ...")
        if p_index < FLAGS.num_adversaries:  # predator
            q_input = tf.concat(obs_ph_n[:FLAGS.num_adversaries] + act_input_n[:FLAGS.num_adversaries], 1)
            train_obs_input = obs_ph_n[:FLAGS.num_adversaries]
            train_action_input = act_ph_n[:FLAGS.num_adversaries]
        else:
            q_input = tf.concat(obs_ph_n[FLAGS.num_adversaries:] + act_input_n[FLAGS.num_adversaries:], 1)
            train_obs_input = obs_ph_n[FLAGS.num_adversaries:]
            train_action_input = act_ph_n[FLAGS.num_adversaries:]

        q_num_units = FLAGS.num_units_ma  # cell number for maddpg
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
            q_num_units = FLAGS.num_units  # cell number for ddpg

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=q_num_units, layer_norm=layer_norm)[:, 0]
        # pg_loss = -tf.reduce_mean(q * weight_ph)
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + reg_loss
        # loss = pg_loss

        # return
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=[act_sample, determin_act_sample])
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=FLAGS.num_units, layer_norm=layer_norm)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=train_obs_input + train_action_input + [weight_ph],
                           # outputs=[loss, pg_loss, distance, reg_loss],
                           outputs=[],
                           updates=[optimize_expr])

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act,
                                             'act_pdtype': act_pdtype_n[p_index]}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        return_ph = tf.placeholder(tf.float32, [None], name="return")
        dis_2_end_ph = tf.placeholder(tf.float32, [None], name="dis_2_end")
        lambda1_ph = tf.placeholder(tf.float32, shape=[], name='lambda1')
        weight_ph = tf.placeholder(tf.float32, [None], name="important_weight")
        # build q-function input
        if q_index < FLAGS.num_adversaries:  # predator
            q_input = tf.concat(obs_ph_n[:FLAGS.num_adversaries] + act_ph_n[:FLAGS.num_adversaries], 1)
            train_obs_input = obs_ph_n[:FLAGS.num_adversaries]
            train_action_input = act_ph_n[:FLAGS.num_adversaries]
        else:
            q_input = tf.concat(obs_ph_n[FLAGS.num_adversaries:] + act_ph_n[FLAGS.num_adversaries:], 1)
            train_obs_input = obs_ph_n[FLAGS.num_adversaries:]
            train_action_input = act_ph_n[FLAGS.num_adversaries:]

        q_num_units = FLAGS.num_units_ma  # cell number for maddpg
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
            q_num_units = FLAGS.num_units  # cell number for ddpg

        q = q_func(q_input, 1, scope="q_func", num_units=q_num_units, layer_norm=layer_norm)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.lambda2), q_func_vars)

        # TODO: for using prioritized replay buffer, adding weight
        td_0 = target_ph - q
        q_loss_td_0 = -tf.reduce_mean(weight_ph * tf.stop_gradient(td_0) * q)
        q_td_0_loss = tf.reduce_mean(weight_ph * tf.square(td_0))

        # TODO: 这里对正向差异 (R-Q) > 0 做截断
        # mask = tf.where(return_ph - tf.squeeze(q) > 0.0,
        #                 tf.ones_like(return_ph), tf.zeros_like(return_ph))
        # TODO: add dis_2_end: return_confidence_factor
        confidence = tf.pow(FLAGS.return_confidence_factor, dis_2_end_ph)
        # td_n = (return_ph * confidence - q) * mask
        # TODO: add clip here...
        # td_n = tf.clip_by_value(return_ph * confidence - q, 0., 4.) * mask
        td_n = tf.clip_by_value(return_ph * confidence - q, 0., 4.)
        q_loss_monte_carlo = -tf.reduce_mean(weight_ph * tf.stop_gradient(td_n) * q)
        # q_td_n_loss = tf.reduce_mean(weight_ph * tf.square((return_ph * confidence - q) * mask))
        q_td_n_loss = tf.reduce_mean(weight_ph * tf.square(td_n))

        loss = q_loss_td_0 + lambda1_ph * q_loss_monte_carlo + reg_loss
        # loss = q_td_0_loss + lambda1_ph * q_td_n_loss + lambda2_ph * margin_classification_loss + reg_loss

        q_values = U.function(train_obs_input + train_action_input, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=q_num_units, layer_norm=layer_norm)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(train_obs_input + train_action_input, target_q)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=train_obs_input + train_action_input + [target_ph] + [weight_ph,
                                                                                        lambda1_ph, dis_2_end_ph,
                                                                                        return_ph,
                                                                                        ],
                           outputs=[],
                           # outputs=[loss, q_loss_td_0, q_loss_monte_carlo, margin_classification_loss, reg_loss,
                           #          q_td_0_loss, q_td_n_loss],
                           updates=[optimize_expr])

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):

    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, buffer,
                 local_q_func=True):
        self.name = name
        self.n = len(obs_shape_n)
        # print('MADDPGAgentTrainer n: ', self.n)
        self._is_deterministic = False
        self.agent_index = agent_index
        # control the important sampling weight
        self.beta = FLAGS.beta

        obs_ph_n = []
        self.mlp_model = model
        self.act_space_n = act_space_n
        # print('MADDPGAgentTrainer act_space_n: ', self.act_space_n)

        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.ddpg_qlr, beta1=0.5, beta2=0.9),
            grad_norm_clipping=0.5,  # 0.5,  # 0.5, 10
            local_q_func=local_q_func,
            layer_norm=False if agent_index < 2 else False
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.ddpg_plr, beta1=0.5, beta2=0.9),
            grad_norm_clipping=0.5,  # 0.5
            local_q_func=local_q_func,
            layer_norm=False if agent_index < 2 else False
        )

        # Create experience buffer
        self.replay_buffer = buffer
        # Coefficient of n-step return
        self.lambda1 = FLAGS.lambda1

        self.running_episode = []
        self.replay_sample_index = None
        print('MADDPGAgentTrainer {} built success...'.format(self.agent_index))

    def toggle_deterministic(self):
        # print("before: ", self._is_deterministic)
        self._is_deterministic = not self._is_deterministic
        # print("after: ", self._is_deterministic)

    def get_actions(self, observations, single=False):
        if single:
            return self.act(observations)[1][0] if self._is_deterministic else self.act(observations)[0][0]
        else:
            return self.act(observations)[1] if self._is_deterministic else self.act(observations)[0]

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.running_episode.append([obs, act, rew, new_obs, done])
        if done:
            add_episode(self.replay_buffer, self.running_episode, gamma=FLAGS.gamma)
            self.running_episode = []

    def preupdate(self):
        self.replay_sample_index = None

    @property
    def pool(self):
        return self.replay_buffer

    def decay_parameters(self):
        # update the parameter which controls the important sampling weight
        # if FLAGS.beta_move:
        #     self.beta = min(self.beta + self.args.beta_increase, 1.)

        # update lambda1 drop
        # self.lambda1 = max(self.lambda1 - self.args.lambda1_decay_rate, self.args.lambda1_min)
        pass

    def do_training(self, agents, iteration, episode):
        self.decay_parameters()

        if not self.is_exploration_enough(FLAGS.min_buffer_size):
            return
        if iteration % (FLAGS.max_episode_len) != 0:
            return

        if FLAGS.prioritized_er:
            return self.update_prioritized(agents, iteration)
        else:
            return self.update(agents, iteration)

    def is_exploration_enough(self, min_pool_size):
        return len(self.pool) >= min_pool_size

    def update(self, agents, iteration):
        '''
        For update using uniform experience replay (using normal tuple data)
        :param agents:
        :param iteration:
        :return:
        '''
        self.replay_sample_index = self.replay_buffer.make_index(FLAGS.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        idxes = self.replay_sample_index
        if self.agent_index < FLAGS.num_adversaries:  # predator
            begin = 0
            end = FLAGS.num_adversaries
        else:  # prey
            begin = FLAGS.num_adversaries
            end = self.n

        for i in range(begin, end):  # sample from friends experience
            obs, act, rew, obs_next, done, dis_2_end, R = agents[i].replay_buffer.sample_index(idxes)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done, dis_2_end, R = self.replay_buffer.sample_index(idxes)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[agent_i].p_debug['target_act'](obs_next_n[obs_i]) for agent_i, obs_i in
                                 zip(range(begin, end), range(end - begin))]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + FLAGS.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        # TODO: checking: W_placeholder=1 when PER is not used.
        q_loss = self.q_train(*(obs_n + act_n + [target_q] + [np.ones_like(target_q),
                                                              0,
                                                              dis_2_end,
                                                              R,
                                                              ]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n + [
            np.ones_like(target_q),
        ]))

        if iteration % FLAGS.target_update_interval == 0:
            self.p_update()
            self.q_update()

        # print("update uniform...")
        return [q_loss, p_loss], None

    def update_prioritized(self, agents, iteration):
        '''
        For update using prioritized experience replay (Using normal tuple data)
        :param agents:
        :param iteration:
        :return:
        '''
        # normal update
        self.replay_sample_index = self.replay_buffer.make_index(FLAGS.batch_size)

        # collect replay sample from all agents
        obs_n, obs_next_n, act_n = [], [], []
        indexes = self.replay_sample_index
        if self.agent_index < FLAGS.num_adversaries:  # predator
            begin = 0
            end = FLAGS.num_adversaries
        else:  # prey
            begin = FLAGS.num_adversaries
            end = self.n

        for i in range(begin, end):  # sample from friends experience
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, idxes = agents[i].replay_buffer.sample_index(
                indexes,
                beta=agents[i].beta)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done, dis_2_end, returns, weights, idxes = self.replay_buffer.sample_index(
            indexes,
            beta=self.beta)

        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[agent_i].p_debug['target_act'](obs_next_n[obs_i]) for agent_i, obs_i in
                                 zip(range(begin, end), range(end - begin))]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + FLAGS.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample

        # positive update
        q_losses = self.q_train(*(obs_n + act_n + [target_q] +
                                  [weights,
                                   0,
                                   dis_2_end,
                                   returns,
                                   ]))

        # train p network.
        p_losses = self.p_train(*(obs_n + act_n + [weights]))

        # update the priority of the replay buffer.
        q_values = self.q_debug['q_values'](*(obs_n + act_n))

        # if self.args.priority_type == 0:
        #     priority = abs(target_q - q_values) + 1e-6
        # elif self.args.priority_type == 1:
        #     priority = abs(returns - q_values) + 1e-6
        # elif self.args.priority_type == 2:
        #     priority = np.maximum(returns - q_values, 1e-6)
        # elif self.args.priority_type == 3:
        #     priority = np.maximum(returns, 1e-6)
        # else:
        priority = abs(target_q - q_values) + 1e-6

        self.replay_buffer.update_priorities(idxes, priority)

        # update the target network.
        if iteration % FLAGS.target_update_interval == 0:
            self.p_update()
            self.q_update()

        # print("update prioritized...")
        return [q_losses, p_losses], None

    def get_session(self):
        return tf.get_default_session()
