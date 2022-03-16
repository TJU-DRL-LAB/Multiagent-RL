# -*- coding: utf-8 -*-
import os

import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用 GPU 0
import tensorflow as tf
import numpy as np
import algorithm.common.tf_utils as tf_utils
import time
import random

# read input cmd from standard input device
flags = tf.app.flags

# Game parameter
flags.DEFINE_string('env_name', 'predator_prey', 'env used')
flags.DEFINE_bool('render', True, 'whether to render the scenario')
flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('num_adversaries', 2, 'num_adversaries')
flags.DEFINE_integer('num_good_agents', 3, 'num_good_agents')
flags.DEFINE_integer('max_step_before_punishment', 8, 'max_step_before_punishment')
flags.DEFINE_bool('reload_prey', True, 'whether to reload the pre-trained prey model')
flags.DEFINE_string('prey_model_path', './model/unit_num_32/model.ckpt-40000',
                    'path of the pre-trained prey model')
flags.DEFINE_bool('reload_predator', False, 'whether to reload the pre-trained predator model')
flags.DEFINE_string('predator_model_path', './your_trained_model_path',
                    'path of the pre-trained predator model')

# Training parameters
flags.DEFINE_bool('learning', True, 'train the agents')
flags.DEFINE_string('predator_policy', 'gasil', 'predator_policy')
flags.DEFINE_string('prey_policy', 'ddpg', 'prey_policy: [random, fixed, ddpg]')
flags.DEFINE_integer('episodes', 140000, 'maximum training episode')
flags.DEFINE_integer('max_episode_len', 60, 'maximum step of each episode')
flags.DEFINE_float('ddpg_plr', 0.01, 'policy learning rate')
flags.DEFINE_float('ddpg_qlr', 0.001, 'critic learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('tau', 0.01, 'target network update frequency')
flags.DEFINE_integer('target_update_interval', 1, 'target network update frequency')
flags.DEFINE_float('return_confidence_factor', 0.7, 'return_confidence_factor')
flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_integer('n_train_repeat', 1, 'repeated sample times at each training time')
flags.DEFINE_integer('save_checkpoint_every_epoch', 5000, 'save_checkpoint_every_epoch')
flags.DEFINE_integer('plot_reward_recent_mean', 1000, 'show the avg reward of recent 200 episode')
flags.DEFINE_bool('save_return', True, 'save trajectory Return by default')
flags.DEFINE_float('lambda1', 0., 'n-step return')
flags.DEFINE_float('lambda1_max', 1., 'n-step return')
flags.DEFINE_float('lambda2', 1e-6, 'coefficient of regularization')

# GASIL
flags.DEFINE_bool('consider_state_action_confidence', True, 'The closer (state, action) to the end state the important')
flags.DEFINE_float('state_action_confidence', 0.8, 'discount factor of (state, action)')
flags.DEFINE_float('state_action_confidence_max', 1., 'discount factor of (state, action)')
flags.DEFINE_integer('gradually_inc_start_episode', 0,
                     'increase parameters start at ${gradually_inc_start_episode} episode')
flags.DEFINE_integer('gradually_inc_within_episode', 12000,
                     'increase parameters in ${gradually_inc_within_episode} episode')
flags.DEFINE_integer('inc_or_dec_step', 1000, 'natural_exp_inc parameter: inc_step')
flags.DEFINE_float('d_lr', 0.001, 'discriminator learning rate')
flags.DEFINE_float('imitation_lambda', 0., 'coefficient of imitation learning')
flags.DEFINE_float('imitation_lambda_max', 1., 'maximum coefficient of imitation learning')
flags.DEFINE_integer('train_discriminator_k', 1, 'train discriminator net k times at each update')
flags.DEFINE_integer('gan_batch_size', 8, 'batch_size of training GAN')

# experience replay
flags.DEFINE_integer('buffer_size', 300000, 'buffer size')
flags.DEFINE_integer('min_buffer_size', 30000, 'minimum buffer size before training')
flags.DEFINE_integer('positive_buffer_size', 32, 'buffer size')
flags.DEFINE_integer('min_positive_buffer_size', 32, 'min buffer size before training')
# prioritized
flags.DEFINE_bool('prioritized_er', False, 'whether to use prioritized ER')
flags.DEFINE_float('alpha', 0.6, 'how much prioritization is used (0 - no prioritization, 1 - full prioritization)')
flags.DEFINE_float('beta', 0.4, 'To what degree to use importance weights (0 - no corrections, 1 - full correction)')

# Net structure
flags.DEFINE_integer('num_units', 32, 'layer neuron number')
flags.DEFINE_integer('num_units_ma', 64, 'layer neuron number for multiagent alg')
flags.DEFINE_integer('h_layer_num', 2, 'hidden layer num')

# Model saving dir
flags.DEFINE_string('model_save_dir', './exp_result/{}/{}/saved_models/seed_{}/model',
                    'Model saving dir')
flags.DEFINE_string('learning_curve_dir', './exp_result/{}/{}/learning_curves/seed_{}',
                    'learning_curve_dir')
FLAGS = flags.FLAGS  # alias


def make_env(scenario_name, max_step_before_punishment):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from env.multiagent.environment import MultiAgentEnv
    import env.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    scenario.max_step_before_punishment = max_step_before_punishment
    print('==============================================================')
    print('max_step_before_punishment: ', scenario.max_step_before_punishment)
    print('==============================================================')

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        info_callback=scenario.collision_number,
                        done_callback=scenario.done,
                        other_callbacks=[scenario.set_arrested_pressed_watched])
    return env


def build_agents(action_dim_n, observation_dim_n, policies_name):
    '''
    build agents
    :param action_dim_n:
    :param observation_dim_n:
    :param policies_name:
    :return:
    '''
    from algorithm.trainer import SimpleAgentFactory
    agents = []
    obs_shape_n = [[dim] for dim in observation_dim_n]
    for agent_idx, policy_name in enumerate(policies_name):
        agents.append(SimpleAgentFactory.createAgent(agent_idx, policy_name, obs_shape_n, action_dim_n, FLAGS))
    return agents


def reload_previous_models(session, env):
    import gc
    # 加载提前训练好的 prey 策略
    if FLAGS.reload_prey:
        prey_vars = []
        for idx in range(FLAGS.num_adversaries, env.n):
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            prey_vars += var

        saver_prey = tf.train.Saver(var_list=prey_vars)
        saver_prey.restore(session, FLAGS.prey_model_path)

        print('[prey] successfully reload previously saved ddpg model({})...'.format(FLAGS.prey_model_path))
        del saver_prey
        gc.collect()
        # all the predator using the same policy
        # best_agent = agents[base_kwargs['num_adversaries']]
        # for i in range(base_kwargs['num_adversaries'], env.n):
        #     agents[i] = best_agent

    if FLAGS.reload_predator:
        predator_vars = []
        for idx in range(FLAGS.num_adversaries):
            var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
            predator_vars += var
        saver_predator = tf.train.Saver(var_list=predator_vars)
        saver_predator.restore(session, FLAGS.predator_model_path)
        print('[predator] successfully reload previously saved ddpg model({})...'.format(
            FLAGS.predator_model_path
        ))
        del saver_predator
        gc.collect()


def train():
    # init env
    env = make_env(FLAGS.env_name, FLAGS.max_step_before_punishment)
    env = env.unwrapped
    # set env seed
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    print("Using seed {} ...".format(FLAGS.seed))

    print('There are total {} agents.'.format(env.n))
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    action_shpe_n = [2] * env.n
    print('obs_shape_n: ', obs_shape_n)  # [16, 16, 16, 14]
    print(action_shpe_n)  # [5, 5, 5, 5]

    adv_policies = [FLAGS.predator_policy] * FLAGS.num_adversaries
    good_policies = [FLAGS.prey_policy] * FLAGS.num_good_agents
    print(adv_policies + good_policies)

    with tf_utils.make_session().as_default() as sess:
        # init agents
        agents = build_agents(action_shpe_n, obs_shape_n, adv_policies + good_policies)

        # init tf summaries
        summary_path = FLAGS.learning_curve_dir.format(FLAGS.env_name, FLAGS.predator_policy, FLAGS.seed)
        print('summary_path', summary_path)
        summary_writer = tf.summary.FileWriter(summary_path)
        adv_mean_options, adv_mean_phs = [], []
        adv_episode_options, adv_episode_phs = [], []
        prey_mean_options, prey_mean_phs = [], []
        prey_episode_options, prey_episode_phs = [], []
        agent0_pool_mean_return_ph = tf.placeholder(dtype=tf.float32, shape=[], name='agent0_pool_mean_return')
        agent0_pool_mean_return_option = tf.summary.scalar('agent0_pool_mean_return', agent0_pool_mean_return_ph)
        agent1_pool_mean_return_ph = tf.placeholder(dtype=tf.float32, shape=[], name='agent1_pool_mean_return')
        agent1_pool_mean_return_option = tf.summary.scalar('agent1_pool_mean_return', agent1_pool_mean_return_ph)
        agent0_positive_pool_mean_return_ph = tf.placeholder(dtype=tf.float32, shape=[],
                                                             name='agent0_positive_pool_mean_return')
        agent0_positive_pool_mean_return_option = tf.summary.scalar('agent0_positive_pool_mean_return',
                                                                    agent0_positive_pool_mean_return_ph)
        agent1_positive_pool_mean_return_ph = tf.placeholder(dtype=tf.float32, shape=[],
                                                             name='agent1_positive_pool_mean_return')
        agent1_positive_pool_mean_return_option = tf.summary.scalar('agent1_positive_pool_mean_return',
                                                                    agent1_positive_pool_mean_return_ph)
        agent0_imitation_lambda = tf.placeholder(dtype=tf.float32, shape=[], name='agent0_imitation_lambda')
        agent0_imitation_lambda_option = tf.summary.scalar('agent0_imitation_lambda', agent0_imitation_lambda)
        agent0_state_action_confidence = tf.placeholder(dtype=tf.float32, shape=[],
                                                        name='agent0_state_action_confidence')
        agent0_state_action_confidence_option = tf.summary.scalar('agent0_state_action_confidence',
                                                                  agent0_state_action_confidence)

        for idx in range(FLAGS.num_adversaries):
            ad_fp_reward_1000_mean = tf.placeholder(dtype=tf.float32, shape=[],
                                                    name='ad_{}_fp_reward_{}_mean'.format(idx,
                                                                                          FLAGS.plot_reward_recent_mean))
            ad_fp_reward_1000_mean_op = tf.summary.scalar(
                'adversary {} episode reward {} mean'.format(idx, FLAGS.plot_reward_recent_mean),
                ad_fp_reward_1000_mean)
            ad_fp_reward_episode = tf.placeholder(dtype=tf.float32, shape=[],
                                                  name='ad_{}_fp_reward_episode'.format(idx))
            ad_fp_reward_episode_op = tf.summary.scalar('adversary {} episode reward'.format(idx), ad_fp_reward_episode)
            adv_mean_phs.append(ad_fp_reward_1000_mean)
            adv_mean_options.append(ad_fp_reward_1000_mean_op)
            adv_episode_phs.append(ad_fp_reward_episode)
            adv_episode_options.append(ad_fp_reward_episode_op)
        for idx in range(FLAGS.num_good_agents):
            prey_fp_reward_1000_mean = tf.placeholder(dtype=tf.float32, shape=[],
                                                      name='prey_{}_fp_reward_{}_mean'.format(idx,
                                                                                              FLAGS.plot_reward_recent_mean))
            prey_fp_reward_1000_mean_op = tf.summary.scalar(
                'prey {} episode reward {} mean'.format(idx, FLAGS.plot_reward_recent_mean),
                prey_fp_reward_1000_mean)
            prey_fp_reward_episode = tf.placeholder(dtype=tf.float32, shape=[],
                                                    name='prey_{}_fp_reward_episode'.format(idx))
            prey_fp_reward_episode_op = tf.summary.scalar('prey {} episode reward'.format(idx), prey_fp_reward_episode)
            prey_mean_phs.append(prey_fp_reward_1000_mean)
            prey_mean_options.append(prey_fp_reward_1000_mean_op)
            prey_episode_phs.append(prey_fp_reward_episode)
            prey_episode_options.append(prey_fp_reward_episode_op)

        # build model saver
        saver = tf.train.Saver(max_to_keep=int(FLAGS.episodes / FLAGS.save_checkpoint_every_epoch))

        # reload previous prey and predator model
        reload_previous_models(session=sess, env=env)

        # Initialize uninitialized variables.
        tf_utils.initialize(sess=sess)
        # assert using same session
        same_session(sess, agents)
        #  make the tensor graph unchangeable
        sess.graph.finalize()

        # collect some statistical data
        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_episode_rewards = [[0.0] for _ in range(env.n)]
        losses_transformation = [[] for _ in range(FLAGS.num_adversaries)]
        coordination_reach_times = [[0] for _ in range(FLAGS.num_good_agents)]
        coordination_times = 0
        miss_coordination_times = 0
        total_times = 0

        obs_n = env.reset()
        episode_step = 0  # step for each episode
        train_step = 0  # total training step
        t_start = time.time()
        print('Starting iterations...')
        while len(episode_rewards) <= FLAGS.episodes:
            # increment global step counter
            train_step += 1

            if FLAGS.render:
                time.sleep(0.3)
                env.render()
            action_2_dim_n = [agent.get_actions(observations=[obs], single=True) for agent, obs in zip(agents, obs_n)]
            action_n = [[0, a[0], 0, a[1], 0] for a in action_2_dim_n]

            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n, restrict_move=True)

            info_n = info_n['n']
            episode_step += 1

            done = all(done_n)  # 达到任务
            terminal = (episode_step >= FLAGS.max_episode_len)  # 最大步数
            ended = done or terminal

            # collect experience
            if FLAGS.learning:
                for i, agent in enumerate(agents):
                    # prey is fixed
                    if i < FLAGS.num_adversaries:
                        agent.experience(obs_n[i], action_2_dim_n[i], rew_n[i], new_obs_n[i], ended)

            # step forward observations
            obs_n = new_obs_n

            # TODO: 这里记录每一轮最大reward
            # record some analysis information
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_episode_rewards[i].append(rew)
                agent_rewards[i][-1] += rew
            for discrete_action in range(FLAGS.num_good_agents):
                coordination_reach_times[discrete_action][-1] += info_n[0][discrete_action]

            # add some log and records...
            if ended:
                # print log for debugging......
                if len(episode_rewards) % 100 == 0:
                    print('process {}, episode {}: '.format(os.getpid(), len(episode_rewards)))

                # reset environment
                obs_n = env.reset()
                # reset episode tags
                episode_step = 0
                episode_rewards.append(0)  # reset sum rewards
                for idx, a in enumerate(agent_rewards):  # reset each agent's reward
                    a.append(0)
                agent_episode_rewards = [[0.0] for _ in range(env.n)]
                for coord_count in coordination_reach_times:  # reset coordination times
                    coord_count.append(0)

            # do training
            if FLAGS.learning:
                for i in range(FLAGS.n_train_repeat):
                    loss_and_positive_loss, trained = [], False
                    for idx, agent in enumerate(agents):
                        if idx >= FLAGS.num_adversaries:
                            continue
                        loss = agent.do_training(agents=agents, iteration=train_step, episode=len(episode_rewards))
                        loss_and_positive_loss.append(loss)
                        trained = loss is not None

                if trained:
                    for idx in range(FLAGS.num_adversaries):
                        losses_transformation[idx].append(loss_and_positive_loss[idx])

                # add summary
                if ended and len(episode_rewards) % 10 == 0:
                    # agent buffer avg return
                    summary_writer.add_summary(sess.run(agent0_pool_mean_return_option,
                                                        {agent0_pool_mean_return_ph: agents[
                                                            0].pool.current_mean_return}),
                                               len(episode_rewards))

                    summary_writer.add_summary(sess.run(agent1_pool_mean_return_option,
                                                        {agent1_pool_mean_return_ph: agents[
                                                            1].pool.current_mean_return}),
                                               len(episode_rewards))

                    if FLAGS.predator_policy == 'gasil':
                        # log confidence_factor,
                        summary_writer.add_summary(
                            sess.run(agent0_state_action_confidence_option,
                                     {agent0_state_action_confidence: agents[0].state_action_confidence}),
                            len(episode_rewards))
                        # imitation_lambda
                        summary_writer.add_summary(
                            sess.run(agent0_imitation_lambda_option,
                                     {agent0_imitation_lambda: agents[0].imitation_lambda}), len(episode_rewards))

                        # positive pool mean return
                        summary_writer.add_summary(sess.run(agent0_positive_pool_mean_return_option,
                                                            {agent0_positive_pool_mean_return_ph: agents[
                                                                0].positive_pool.current_mean_return}),
                                                   len(episode_rewards))

                        summary_writer.add_summary(sess.run(agent1_positive_pool_mean_return_option,
                                                            {agent1_positive_pool_mean_return_ph: agents[
                                                                1].positive_pool.current_mean_return}),
                                                   len(episode_rewards))

                    for idx in range(FLAGS.num_adversaries):
                        summary_writer.add_summary(sess.run(adv_mean_options[idx], {
                            adv_mean_phs[idx]: np.mean(
                                agent_rewards[idx][-FLAGS.plot_reward_recent_mean - 1: -1])}),
                                                   len(episode_rewards))
                        summary_writer.add_summary(
                            sess.run(adv_episode_options[idx], {adv_episode_phs[idx]: agent_rewards[idx][-2]}),
                            len(episode_rewards))

                    # add summary for drawing curves (prey)
                    for idx in range(FLAGS.num_good_agents):
                        summary_writer.add_summary(sess.run(prey_mean_options[idx], {
                            prey_mean_phs[idx]: np.mean(
                                agent_rewards[idx + FLAGS.num_adversaries][
                                -FLAGS.plot_reward_recent_mean - 1: -1])
                        }), len(episode_rewards))
                        summary_writer.add_summary(
                            sess.run(prey_episode_options[idx], {
                                prey_episode_phs[idx]: agent_rewards[idx + FLAGS.num_adversaries][-2]
                            }), len(episode_rewards))

                # save models
                if ended and len(episode_rewards) % FLAGS.save_checkpoint_every_epoch == 0:
                    # save model
                    save_model(saver, sess, len(episode_rewards))

                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards),
                        np.mean(episode_rewards[-FLAGS.save_checkpoint_every_epoch:]),
                        [np.mean(rew[-FLAGS.save_checkpoint_every_epoch:]) for rew in agent_rewards],
                        round(time.time() - t_start, 3)))
                    t_start = time.time()

        # close sess
        sess.close()

        # record
        if FLAGS.learning:
            record_logs(**{
                'summary_path': summary_path,
                'agent_rewards': agent_rewards,
                'coordination_reach_times': coordination_reach_times,
                'agents': agents,
                'losses_transformation': losses_transformation,
            })


def save_model(saver, sess, episode):
    model_path = FLAGS.model_save_dir.format(FLAGS.env_name, FLAGS.predator_policy, FLAGS.seed)
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    saver.save(sess, model_path, global_step=episode)


def record_logs(**kwargs):
    log_path = kwargs['summary_path'] + '/logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    losses_transformation_file_name = log_path + '/' + FLAGS.env_name + '_losses_transformation.pkl'
    with open(losses_transformation_file_name, 'wb') as fp:
        pickle.dump(kwargs['losses_transformation'], fp)

    rew_file_name = log_path + '/' + FLAGS.env_name + '_rewards.pkl'
    with open(rew_file_name, 'wb') as fp:
        pickle.dump(kwargs['agent_rewards'], fp)

    coordination_reach_times_file = log_path + '/' + FLAGS.env_name + '_coordination_reach_times.pkl'
    with open(coordination_reach_times_file, 'wb') as fp:
        pickle.dump(kwargs['coordination_reach_times'], fp)

    buffer_mean = log_path + '/' + FLAGS.env_name + '_buffer.pkl'
    with open(buffer_mean, 'wb') as fp:
        pickle.dump(kwargs['agents'][0].pool.mean_returns, fp)
    print("Mean buffer return:")
    print(kwargs['agents'][0].pool.mean_returns)


# for debug below ..........................................................
def same_session(sess, agents):
    for agent in agents[:FLAGS.num_adversaries]:
        if sess != agent.get_session():
            print("Session error (diff tf session)")
    print("The same session.........................")


if __name__ == '__main__':
    train()
