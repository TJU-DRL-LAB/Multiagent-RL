#!/usr/bin/env python3
# encoding=utf-8

import os
# import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../')

from algs.hDQN_128 import DeepQNetwork as DQN
from envs.MATC_2a2d8t_Ring.MATC_Ring_non_hie import GameEnv

# def render(render, pause, is_block=False):
#     plt.imshow(render)
#     plt.show(block=is_block)
#     plt.pause(pause)
#     plt.clf()

# --------------------------------- Param parser ------------------------------
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--mem', type=int, default=10000)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--before-learning', type=float, default=50000)
parser.add_argument('--train-interval', type=int, default=20)
parser.add_argument('--GPU-divide', type=float, default=0.03)
parser.add_argument('--GPU-no', type=str, default='3')
parser.add_argument('--soft-replace', type=bool, default=False)
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--logs-path', type=str, default='./tf_logs/IL/')
parser.add_argument('--e', type=float, default=0.1)
parser.add_argument('--ince', type=float, default=0.00001)
parser.add_argument('--max-timestep', type=int, default=100)


args = parser.parse_args()
print("Params:")
print("--seed:", args.seed)
print("--mem:", args.mem)
print("--batch:", args.batch)
print("--lr:", args.lr)
print("--gamma:", args.gamma)
print("--before-learning:", args.before_learning)
print("--train-interval:", args.train_interval)
print("--GPU-divide:", args.GPU_divide)
print("--GPU-no:", args.GPU_no)
print("--soft-replace:", args.soft_replace)
print("--logs-path:", args.logs_path)
print("--e:", args.e)
print("--ince:", args.ince)
print("--max-timestep:", args.max_timestep)
print("--")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_no

seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

MAX_EPISODE = 100000
MAX_TIMESTEP = args.max_timestep

mem = args.mem
batch = args.batch
lr = args.lr
gamma = args.gamma
before_learning = args.before_learning
train_interval = args.train_interval

is_soft_replace = args.soft_replace
logs_path = args.logs_path

game_size = 11
is_fixed = True

IS_SAVE_MODEL = False
IS_SAVE_LOG = True
# IS_SAVE_LOG = False

action_num = 7

env = GameEnv(width=game_size, height=game_size, is_fixed=is_fixed)
# s = env.reset()

# --------------------------------- GPU config ------------------------------
GPU_divide = args.GPU_divide

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_divide
sess = tf.Session(config=config)

dqn1 = DQN(n_features=game_size*game_size*3 + 1,
           n_actions=action_num,
           learning_rate=lr,
           reward_decay=gamma,
           replace_target_iter=100,
           memory_size=mem,
           batch_size=batch,
           e_greedy_increment=args.ince,
           session=sess,
           mark='1',
           soft_replace=is_soft_replace,
           e_greedy=args.e
           )

dqn2 = DQN(n_features=game_size*game_size*3 + 1,
           n_actions=action_num,
           learning_rate=lr,
           reward_decay=gamma,
           replace_target_iter=100,
           memory_size=mem,
           batch_size=batch,
           e_greedy_increment=args.ince,
           session=sess,
           mark='2',
           soft_replace=is_soft_replace,
           e_greedy=args.e
           )

# --------------------------------- summary operations ------------------------------
if IS_SAVE_LOG:
    var1 = tf.placeholder(tf.float32, [None, ], name='avg_r')
    var2 = tf.placeholder(tf.float32, [None, ], name='avg_r')
    succ_100 = tf.placeholder(tf.float32, [None, ], name='succ_rate_100')
    succ_1000 = tf.placeholder(tf.float32, [None, ], name='succ_rate_1000')
    step_100 = tf.placeholder(tf.float32, [None, ], name='finish_steps_100')
    a1_avg_r_100 = tf.reduce_mean(var1)
    a2_avg_r_100 = tf.reduce_mean(var2)
    fs_100 = tf.reduce_mean(step_100)
    sc_100 = tf.reduce_mean(succ_100)
    sc_1k = tf.reduce_mean(succ_1000)
    total_r = a1_avg_r_100 + a2_avg_r_100
    res_r = tf.abs(a1_avg_r_100 - a2_avg_r_100)
    tf.summary.scalar('a1_avg_r_100', a1_avg_r_100)
    tf.summary.scalar('a2_avg_r_100', a2_avg_r_100)
    tf.summary.scalar('succ_rate_100', sc_100)
    tf.summary.scalar('succ_rate_1000', sc_1k)
    tf.summary.scalar('avg_r_sum', total_r)
    tf.summary.scalar('avg_r_res', res_r)
    tf.summary.scalar('finish_steps_100', fs_100)

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(logs_path + 'g9_lr25_1e51e1_fc128b64m10kv20/seed' + str(seed), sess.graph)
#
sess.run(tf.global_variables_initializer())

# --------------------------------- other params ------------------------------
episode = 0
# done = False
success_count = 0
train_time = 0
total_step = 0
total_goal_step = 0
result_list = [0]

render_episode = 1
max_success_rate_100 = 0
max_success_rate_1000 = 0

print_interval = 500

a1_rs = [0]
a2_rs = [0]
a1_rs_max_100 = -1
a2_rs_max_100 = -1
game_steps = [0]

stop_flag = False

# --------------------------------- training ------------------------------

while episode < MAX_EPISODE:
# while True:
    episode_step = 0
    # episode_goal_step = 0
    s = env.reset()
    ar1 = 0
    ar2 = 0
    done = False

    m1 = 0
    m2 = 0
    while True:
        # if episode >= MAX_EPISODE:
        #     render(render=env.render_env(), pause=0.01, is_block=False)

        a1 = dqn1.choose_action(np.array(s + [m1]))
        a2 = dqn2.choose_action(np.array(s + [m2]))

        s_, m1_, m2_, r1, r2, done = env.step(a1, a2)

        ar1 += r1
        ar2 += r2

        dqn1.store_transition(s + [m1], a1, r1, s_ + [m1_])
        dqn2.store_transition(s + [m2], a2, r2, s_ + [m2_])

        episode_step += 1
        total_step += 1

        # train networks
        if (total_step > before_learning) and (total_step % train_interval == 0):
            if not stop_flag:
                train_time += 1
                dqn1.learn()
                dqn2.learn()

        # break while loop when end of this episode
        if done or episode_step >= MAX_TIMESTEP:
            if done:
                success_count += 1
                result_list.append(1)
            else:
                result_list.append(0)
            a1_rs.append(ar1)
            a2_rs.append(ar2)
            game_steps.append(episode_step)
            break

        # swap observation
        s = s_


    result_list = result_list[-1000:]
    success_rate_recent_100 = sum(result_list[-100:]) / 100.0
    success_rate_recent_1000 = sum(result_list) / 1000.0
    max_success_rate_100 = success_rate_recent_100 if success_rate_recent_100 > max_success_rate_100 else max_success_rate_100
    max_success_rate_1000 = success_rate_recent_1000 if success_rate_recent_1000 > max_success_rate_1000 else max_success_rate_1000

    a1_rs = a1_rs[-100:]
    a2_rs = a2_rs[-100:]
    if episode > 100:
        a1_rs_100 = sum(a1_rs) / 100.0
        a2_rs_100 = sum(a2_rs) / 100.0
    else:
        a1_rs_100 = a1_rs[-1]
        a2_rs_100 = a2_rs[-1]
    a1_rs_max_100 = a1_rs_100 if a1_rs_100 > a1_rs_max_100 else a1_rs_max_100
    a2_rs_max_100 = a2_rs_100 if a2_rs_100 > a2_rs_max_100 else a2_rs_max_100

    # TODO
    game_steps = game_steps[-100:]
    game_steps_100 = sum(game_steps) / 100.0

    episode += 1
    # print('--Episode:', episode, '. AR:', ar)

    # TODO
    if IS_SAVE_LOG:
        summary = sess.run(merged, feed_dict={var1: a1_rs,
                                              var2: a2_rs,
                                              succ_1000: result_list,
                                              succ_100: result_list[-100:],
                                              step_100: game_steps})
        writer.add_summary(summary, episode)

    if episode % print_interval == 0:
        print('--Episode:', episode, '. Train time:', train_time,
              '. Succ_rate:', success_rate_recent_100, '/', max_success_rate_100, ',',
              success_rate_recent_1000, '/', max_success_rate_1000,
              '. Accm_r: %.2f' % a1_rs_100, '(%.2f)' % a1_rs_max_100, '/ %.2f' % a2_rs_100, '(%.2f)' % a2_rs_max_100,
              '. Steps: %.2f' % game_steps_100)


# --------------------------------- playing ------------------------------

print('---')
print('END')

