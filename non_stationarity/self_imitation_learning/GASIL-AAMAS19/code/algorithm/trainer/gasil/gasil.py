import numpy as np
import tensorflow as tf

import algorithm.common.tf_utils as U
from algorithm.trainer import AgentTrainer
from algorithm.common.distributions2 import make_pdtype
from algorithm.common.reinforce_utils import make_update_exp, natural_exp_inc
from algorithm.prioritized_experience_replay_buffer.utils import add_episode, add_trajectory

FLAGS = tf.app.flags.FLAGS  # alias

def p_train(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False,
            scope="trainer", reuse=None, p_layer_norm=False, q_layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        # weight_ph = tf.placeholder(tf.float32, [None], name="important_weight")
        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=FLAGS.num_units,
                   layer_norm=p_layer_norm)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        reg_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(FLAGS.lambda2), p_func_vars)

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        # TODO: 这里添加了 deterministic
        determin_act_sample, act_sample = act_pd.sample(deterministic=True)
        # p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()  # act_pd.mode() sample action from current policy

        # build q-function input
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

        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=q_num_units, layer_norm=q_layer_norm)[:, 0]
        # pg_loss = -tf.reduce_mean(q * weight_ph)
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + reg_loss
        # loss = pg_loss

        # return
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=[act_sample, determin_act_sample])
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=FLAGS.num_units, layer_norm=p_layer_norm)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        # Create callable functions
        # train = U.function(inputs=train_obs_input + train_action_input + [weight_ph],
        train = U.function(inputs=train_obs_input + train_action_input,
                           # outputs=[loss, pg_loss, distance, reg_loss],
                           outputs=[],
                           updates=[optimize_expr])
        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act,
                                             'act_pdtype': act_pdtype_n[p_index]}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None,
            local_q_func=False, scope="trainer", reuse=None, q_layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        # for td-n
        return_ph = tf.placeholder(tf.float32, [None], name="return")
        dis_2_end_ph = tf.placeholder(tf.float32, [None], name="dis_2_end")
        lambda1_ph = tf.placeholder(tf.float32, shape=[], name='lambda1_n_step')
        # for important sampling
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

        q = q_func(q_input, 1, scope="q_func", num_units=q_num_units, layer_norm=q_layer_norm)[:, 0]
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
        # q_td_n_loss = tf.reduce_mean(weight_ph * tf.square(td_n))

        loss = q_loss_td_0 + lambda1_ph * q_loss_monte_carlo + reg_loss

        q_values = U.function(train_obs_input + train_action_input, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=q_num_units, layer_norm=q_layer_norm)[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(train_obs_input + train_action_input, target_q)

        # build optimizer
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(
            inputs=train_obs_input + train_action_input + [target_ph, weight_ph, lambda1_ph, dis_2_end_ph, return_ph],
            outputs=[],
            # outputs=[loss, q_loss_td_0, q_loss_monte_carlo, margin_classification_loss, reg_loss,
            #          q_td_0_loss, q_td_n_loss],
            updates=[optimize_expr])

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


def discriminator_train(obs_shape_n, act_space_n, agent_index, discriminator_func, optimizer, grad_norm_clipping=None,
                        scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        expert_dis_2_end_ph = tf.placeholder(tf.float32, [None], name="expert_dis_2_end")
        policy_dis_2_end_ph = tf.placeholder(tf.float32, [None], name="policy_dis_2_end")
        state_action_confidence_factor_ph = tf.placeholder(tf.float32, shape=[], name='state_action_confidence_factor')

        # create act distributions
        act_pdtype = make_pdtype(act_space_n[agent_index])
        # set up placeholders
        expert_act_ph = act_pdtype.sample_placeholder([None], name="expert_action" + str(agent_index))
        expert_state_ph = U.BatchInput(obs_shape_n[agent_index], name="expert_observation" + str(agent_index)).get()

        policy_act_ph = act_pdtype.sample_placeholder([None], name="policy_action" + str(agent_index))
        policy_state_ph = U.BatchInput(obs_shape_n[agent_index], name="policy_observation" + str(agent_index)).get()

        # input for discriminator
        expert_input = tf.concat([expert_state_ph, expert_act_ph], 1)
        d_model_real, d_logits_real = discriminator_func(expert_input, scope="discriminator",
                                                         num_units=FLAGS.num_units)
        policy_input = tf.concat([policy_state_ph, policy_act_ph], 1)
        d_model_fake, d_logits_fake = discriminator_func(policy_input, scope="discriminator", reuse=True,
                                                         num_units=FLAGS.num_units)

        discriminator_func_vars = U.scope_vars(U.absolute_scope_name("discriminator"))

        # Calculate losses
        # To help the discriminator generalize better, the labels are reduced a bit from 1.0 to 0.9,
        # for example, using the parameter smooth. This is known as label smoothing, typically used with classifiers
        # to improve performance.
        smooth = 0.1
        if FLAGS.consider_state_action_confidence:
            print('consider state action confidence...')
            expert_confidence = tf.pow(state_action_confidence_factor_ph, expert_dis_2_end_ph)
            expert_confidence_sum = tf.reduce_sum(expert_confidence)
            d_loss_real = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                        labels=tf.ones_like(d_logits_real) * (
                                                                1 - smooth)) * expert_confidence) / expert_confidence_sum

            policy_confidence = tf.pow(state_action_confidence_factor_ph, policy_dis_2_end_ph)
            policy_confidence_sum = tf.reduce_sum(policy_confidence)
            d_loss_fake = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                        labels=tf.zeros_like(
                                                            d_logits_fake)) * policy_confidence) / policy_confidence_sum
        else:
            print("doesn't consider state action confidence...")
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                        labels=tf.ones_like(d_logits_real) * (1 - smooth)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                        labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake

        # build optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
        update_ops_q = [item for item in update_ops if item.name.find('discriminator') != -1]
        print('discriminator-func, batch norm update parameters: ', update_ops_q)
        print("all update options: ", tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies(update_ops_q):
            optimize_expr = U.minimize_and_clip(optimizer, d_loss, discriminator_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(
            inputs=[expert_state_ph, expert_act_ph, policy_state_ph, policy_act_ph, state_action_confidence_factor_ph,
                    expert_dis_2_end_ph, policy_dis_2_end_ph],
            outputs=[d_loss, d_loss_fake],
            updates=[optimize_expr])

        # d_model_fake_values = U.function([policy_state_ph, policy_act_ph], outputs=d_model_fake)
        # -np.log(0.99)=0.01, -np.log(0.01)=4.61
        d_model_fake_clipped = tf.clip_by_value(d_model_fake, 0.01, 0.99)
        # d_model_fake_clipped = tf.clip_by_value(d_model_fake, 0.1, 0.9)
        # TODO: clip reward to [-0.5, 1.5]
        imitation_reward = tf.clip_by_value(-tf.log(1. - d_model_fake_clipped)[:, 0] + np.log(0.5), -0.5, 1.5)
        # if args.subtract_baseline:
        #     imitation_reward = -tf.log(1. - d_model_fake_clipped)[:, 0] + np.log(0.5)
        # else:
        #     imitation_reward = -tf.log(1. - d_model_fake_clipped)[:, 0]
        imitation_reward_values = U.function([policy_state_ph, policy_act_ph], outputs=imitation_reward)
        return train, imitation_reward_values


class GASIL_DDPGAgentTrainer(AgentTrainer):

    def __init__(self, name, model, discriminator, obs_shape_n, act_space_n, agent_index, buffer,
                 trajectory_buffer, positive_buffer, local_q_func=True):
        self.name = name
        self.n = len(obs_shape_n)
        # print('MADDPGAgentTrainer n: ', self.n)
        self._is_deterministic = False
        self.agent_index = agent_index

        obs_ph_n = []
        self.mlp_model = model
        self.mlp_discriminator = discriminator
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
            grad_norm_clipping=10,  # ,  # 0.5, 10
            local_q_func=local_q_func,
            q_layer_norm=False if agent_index < 2 else False,
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
            p_layer_norm=False if agent_index < 2 else False,
            q_layer_norm=False if agent_index < 2 else False
        )

        self.discriminator_train, self.imitation_reward_values = discriminator_train(
            obs_shape_n,
            act_space_n,
            agent_index,
            discriminator,
            tf.train.AdamOptimizer(learning_rate=FLAGS.d_lr, beta1=0.5, beta2=0.9),
            grad_norm_clipping=0.5,
            scope=self.name,
        )

        # Create experience buffer
        self.replay_buffer = buffer
        self.trajectory_replay_buffer = trajectory_buffer
        self.positive_replay_buffer = positive_buffer

        self.lambda1 = FLAGS.lambda1
        self.imitation_lambda = FLAGS.imitation_lambda
        self.state_action_confidence = FLAGS.state_action_confidence

        self.running_episode = []
        self.replay_sample_index = None
        print('GASIL_DDPGAgentTrainer {} built success...'.format(self.agent_index))

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
            # add tuples to normal buffer
            add_episode(self.replay_buffer, self.running_episode, gamma=FLAGS.gamma)
            # add a new trajectory to trajectory buffer
            add_trajectory(self.trajectory_replay_buffer, self.running_episode, gamma=FLAGS.gamma)
            # 每一条都加，靠小顶堆自己过滤
            add_trajectory(self.positive_replay_buffer, self.running_episode, gamma=FLAGS.gamma)
            self.running_episode = []

    def preupdate(self):
        self.replay_sample_index = None

    @property
    def pool(self):
        return self.replay_buffer

    @property
    def trajectory_pool(self):
        return self.trajectory_replay_buffer

    @property
    def positive_pool(self):
        return self.positive_replay_buffer

    def inc_parameters(self, global_episode):
        # update state_action_confidence inc
        self.state_action_confidence = natural_exp_inc(init_param=FLAGS.state_action_confidence,
                                                        max_param=FLAGS.state_action_confidence_max,
                                                        global_step=FLAGS.gradually_inc_within_episode,
                                                        current_step=global_episode - FLAGS.gradually_inc_start_episode,
                                                        inc_step=FLAGS.inc_or_dec_step,
                                                        inc_rate=0.5, stair_case=False)


        if global_episode > FLAGS.gradually_inc_start_episode and global_episode < FLAGS.gradually_inc_within_episode:
            # update imitation_lambda inc
            self.imitation_lambda = natural_exp_inc(init_param=FLAGS.imitation_lambda,
                                             max_param=FLAGS.imitation_lambda_max,
                                             global_step = FLAGS.gradually_inc_within_episode,
                                             current_step = global_episode - FLAGS.gradually_inc_start_episode,
                                             inc_step = FLAGS.inc_or_dec_step,
                                             inc_rate=0.5, stair_case=False)
            # update lambda1 inc
            self.lambda1 = natural_exp_inc(init_param=FLAGS.lambda1,
                                           max_param=FLAGS.lambda1_max,
                                           global_step=FLAGS.gradually_inc_within_episode,
                                           current_step=global_episode - FLAGS.gradually_inc_start_episode,
                                           inc_step=FLAGS.inc_or_dec_step,
                                           inc_rate=0.5, stair_case=False)

    def do_training(self, agents, iteration, episode):
        self.inc_parameters(episode)

        if not self.is_exploration_enough(self.replay_buffer, FLAGS.min_buffer_size):
            return
        if iteration % (FLAGS.max_episode_len) != 0:
            return

        losses = self._update_normal(agents)

        # positive update
        positive_losses = None

        if self.is_exploration_enough(self.positive_replay_buffer, FLAGS.min_positive_buffer_size):
            positive_losses = self.update_self_imitation(agents, iteration)

        # update the target network.
        if iteration % FLAGS.target_update_interval == 0:
            self.p_update()
            self.q_update()
        return losses, positive_losses

    def is_exploration_enough(self, pool, min_pool_size):
        return len(pool) >= min_pool_size

    # TODO: 只有在 deterministic reward 下可用，因为只有此时，时刻是对齐的，伙伴之间Reward一样
    adversary_sharing_indexes = None

    def update_self_imitation(self, agents, iteration):
        '''
        For update using prioritized experience replay
        :param agents:
        :param iteration:
        :return:
        '''
        # TODO 1: train discriminator K times：
        discriminator_loss_action_rt = 0
        discriminator_loss_state_action_rt = 0
        generator_indicator_action_rt = 0
        generator_indicator_state_action_rt = 0
        for train_discriminator_k in range(FLAGS.train_discriminator_k):
            # print("positive idx is not sharing...")
            expert_idxes = self.positive_pool.make_index(FLAGS.gan_batch_size)

            expert_obs, expert_act, expert_rew, expert_obs_next, expert_done, expert_dis_2_end, expert_returns, expert_weights, expert_ranges = self.positive_pool.sample_index(
                expert_idxes)

            # 抽最新的 on policy 样本轨迹
            policy_idxes = self.trajectory_pool.make_index(FLAGS.gan_batch_size)
            policy_obs, policy_act, policy_rew, policy_obs_next, policy_done, policy_dis_2_end, policy_returns, policy_weights, policy_ranges = self.trajectory_pool.sample_index(
                policy_idxes)

            # TODO 1.2: train discriminator
            # (1) 首先只模仿 action 分布 (只采用专家样本数据)
            on_policy_act = self.get_actions(expert_obs)
            discriminator_loss_action, generator_indicator_action= self.discriminator_train(
                *[expert_obs, expert_act, expert_obs, on_policy_act, self.state_action_confidence, expert_dis_2_end,
                  expert_dis_2_end])
            discriminator_loss_action_rt += discriminator_loss_action
            generator_indicator_action_rt += generator_indicator_action

            # (2) 模仿 state action 分布 (采用专家跟自己生成的样本)
            discriminator_loss_state_action, generator_indicator_state_action= self.discriminator_train(
                *[expert_obs, expert_act, policy_obs, policy_act, self.state_action_confidence, expert_dis_2_end,
                  policy_dis_2_end])
            discriminator_loss_state_action_rt += discriminator_loss_state_action
            generator_indicator_state_action_rt += generator_indicator_state_action

        discriminator_loss_action_rt /= FLAGS.train_discriminator_k
        discriminator_loss_state_action_rt /= FLAGS.train_discriminator_k
        generator_indicator_action_rt /= FLAGS.train_discriminator_k
        generator_indicator_state_action_rt /= FLAGS.train_discriminator_k

        # TODO 2: cal shaping reward
        # (1) 首先只模仿 action 分布 (只采用专家样本数据) # TODO: 专家样本的self-imitation
        imitation_reward_action = self.imitation_reward_values(*[expert_obs, expert_act])
        # (2) 模仿 state action 分布 (采用专家跟自己生成的样本)
        imitation_reward_state_action = self.imitation_reward_values(*[policy_obs, policy_act])

        # TODO 3: train generator/train policy gradient
        # (1) 首先只模仿 action 分布 (只采用专家样本数据)
        # collect replay sample from all agents
        if self.agent_index < FLAGS.num_adversaries:  # predator
            begin = 0
            end = FLAGS.num_adversaries
        else:  # prey
            begin = FLAGS.num_adversaries
            end = self.n

        obs_n, obs_next_n, act_n = [], [], []
        for i in range(begin, end):  # sample from friends experience
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = agents[i].positive_pool.sample_index(
                expert_idxes)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # train q network
        num_sample, target_q = 1, 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[agent_i].p_debug['target_act'](obs_next_n[obs_i]) for agent_i, obs_i in
                                 zip(range(begin, end), range(end - begin))]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += expert_rew + self.imitation_lambda * imitation_reward_action + FLAGS.gamma * (
                    1.0 - expert_done) * target_q_next
        target_q /= num_sample
        q_losses_action = self.q_train(
            *(obs_n + act_n + [target_q, expert_weights, self.lambda1, expert_dis_2_end, expert_returns]))
        # train p network.
        p_losses_action = self.p_train(*(obs_n + act_n))

        # (2) 模仿 state action 分布 (采用专家跟自己生成的样本)
        # collect replay sample from all agents
        obs_n, obs_next_n, act_n = [], [], []
        for i in range(begin, end):  # sample from friends experience
            obs, act, rew, obs_next, done, dis_2_end, returns, weights, ranges = agents[
                i].trajectory_pool.sample_index(policy_idxes)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # train q network
        num_sample, target_q = 1, 0.0
        for i in range(num_sample):
            target_act_next_n = [agents[agent_i].p_debug['target_act'](obs_next_n[obs_i]) for agent_i, obs_i in
                                 zip(range(begin, end), range(end - begin))]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += policy_rew + self.imitation_lambda * imitation_reward_state_action + FLAGS.gamma * (
                    1.0 - policy_done) * target_q_next
        target_q /= num_sample
        # TODO: 这只是普通轨迹，应该不需要算 n-step return
        q_losses_state_action = self.q_train(
            # *(obs_n + act_n + [target_q, policy_weights, self.lambda1, policy_dis_2_end, policy_returns]))
            *(obs_n + act_n + [target_q, policy_weights, 0, policy_dis_2_end, policy_returns]))
        # train p network.
        p_losses_state_action = self.p_train(*(obs_n + act_n))

        # print("Positive trajectory Prioritized buffer Updated...")
        return [discriminator_loss_action_rt, discriminator_loss_state_action_rt, generator_indicator_action_rt,
                generator_indicator_state_action_rt, np.mean(imitation_reward_action), np.mean(imitation_reward_state_action)], [[q_losses_action, p_losses_action],
                                                       [q_losses_state_action, p_losses_state_action]]

    def _update_normal(self, agents):
        '''
        For update using uniform experience replay (using normal tuple data)
        :param agents:
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

        q_loss = self.q_train(*(obs_n + act_n + [target_q] + [np.ones_like(target_q),
                                                              0,
                                                              dis_2_end,
                                                              R,
                                                              ]))

        # train p network
        p_loss = self.p_train(*(obs_n + act_n))

        # print("Normal trajectory buffer Updated...")
        return [q_loss, p_loss]

    def get_buffer(self, positive):
        if positive:
            return self.positive_pool
        else:
            return self.pool

    def get_session(self):
        return tf.get_default_session()
