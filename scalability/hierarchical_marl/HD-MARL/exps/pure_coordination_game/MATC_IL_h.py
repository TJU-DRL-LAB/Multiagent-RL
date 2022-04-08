#!/usr/bin/env python3
# encoding=utf-8

import os
# import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../')

from algs.hDQN import DeepQNetwork as DQN
from algs.hDQN import metaDeepQNetwork as metaDQN
from envs.classic_coordination_games.MATC_PC import GameEnv

# def render(render, pause, is_block=False):
#     plt.imshow(render)
#     plt.show(block=is_block)
#     plt.pause(pause)
#     plt.clf()

# --------------------------------- Param parser ------------------------------
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--meta-mem', type=int, default=5000)
parser.add_argument('--mem', type=int, default=5000)
parser.add_argument('--meta-batch', type=int, default=64)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--meta-lr', type=float, default=0.00025)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--meta-gamma', type=float, default=0.95)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--before-learning', type=float, default=50000)
parser.add_argument('--train-interval', type=int, default=20)
parser.add_argument('--pre-train', type=int, default=50000)
parser.add_argument('--GPU-divide', type=float, default=0.05)
parser.add_argument('--GPU-no', type=str, default='3')
parser.add_argument('--is-stop', type=bool, default=False)
parser.add_argument('--soft-replace', type=bool, default=False)
parser.add_argument('--seed', type=int, default=111)
parser.add_argument('--logs-path', type=str, default='./tf_logs/IL-h/')
parser.add_argument('--meta-e', type=float, default=0.1)
parser.add_argument('--meta-ince', type=float, default=0.00001)
parser.add_argument('--e', type=float, default=0.1)
parser.add_argument('--ince', type=float, default=0.0001)
parser.add_argument('--subpolicy-len', type=int, default=15)
parser.add_argument('--max-timestep', type=int, default=50)

args = parser.parse_args()
print("Params:")
print("--seed:", args.seed)
# print("--agent-seed:", hDQN.SEED)
# print("--env-seed:", MATC.SEED)
print("--meta-mem:", args.meta_mem)
print("--mem:", args.mem)
print("--meta-batch:", args.meta_batch)
print("--batch:", args.batch)
print("--meta-lr:", args.meta_lr)
print("--lr:", args.lr)
print("--meta-gamma:", args.meta_gamma)
print("--gamma:", args.gamma)
print("--before-learning:", args.before_learning)
print("--train-interval:", args.train_interval)
print("--pre-train:", args.pre_train)
print("--GPU-divide:", args.GPU_divide)
print("--GPU-no:", args.GPU_no)
print("--is-stop:", args.is_stop)
print("--soft-replace:", args.soft_replace)
print("--logs-path:", args.logs_path)
print("--meta-e:", args.meta_e)
print("--meta-ince:", args.meta_ince)
print("--e:", args.e)
print("--ince:", args.ince)
print("--subpolicy-len:", args.subpolicy_len)
print("--max-timestep:", args.max_timestep)
print("--")

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_no

seed = args.seed
np.random.seed(seed)
tf.set_random_seed(seed)

MAX_EPISODE = 100000
MAX_TIMESTEP = args.max_timestep
MAX_GOALSTEP = args.subpolicy_len

meta_mem = args.meta_mem
mem = args.mem
meta_batch = args.meta_batch
batch = args.batch
meta_lr = args.meta_lr
lr = args.lr
meta_gamma = args.meta_gamma
gamma = args.gamma
before_learning = args.before_learning
train_interval = args.train_interval
pre_train = args.pre_train

is_soft_replace = args.soft_replace
logs_path = args.logs_path

game_width = 15
game_height = 7
is_fixed = True

IS_STOP = args.is_stop
IS_SAVE_MODEL = False
# IS_SAVE_LOG = False
IS_SAVE_LOG = True

# 0: go to dump, 1: go to trash 1, 2: go to trash 2, 3: load, 4: place
goal_num = 5
nav_action_num = 6
# nav_action_num = 5

env = GameEnv(width=game_width, height=game_height, is_fixed=is_fixed)
# s = env.reset()

# --------------------------------- GPU config ------------------------------
GPU_divide = args.GPU_divide

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_divide
sess = tf.Session(config=config)

meta_dqn1 = metaDQN(n_features=game_width * game_height * 4 + 1,
                    n_actions=goal_num,
                    learning_rate=meta_lr,
                    reward_decay=meta_gamma,
                    replace_target_iter=100,
                    memory_size=meta_mem,
                    batch_size=meta_batch,
                    e_greedy_increment=args.meta_ince,
                    session=sess,
                    mark='1',
                    soft_replace=is_soft_replace,
                    e_greedy=args.meta_e
                    )

dqn1 = DQN(n_features=game_width * game_height * 5 + 1,
           n_actions=nav_action_num,
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

meta_dqn2 = metaDQN(n_features=game_width * game_height * 4 + 1,
                    n_actions=goal_num,
                    learning_rate=meta_lr,
                    reward_decay=meta_gamma,
                    replace_target_iter=100,
                    memory_size=meta_mem,
                    batch_size=meta_batch,
                    e_greedy_increment=args.meta_ince,
                    session=sess,
                    mark='2',
                    soft_replace=is_soft_replace,
                    e_greedy=args.meta_e
                    )

dqn2 = DQN(n_features=game_width * game_height * 5 + 1,
           n_actions=nav_action_num,
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

    writer = tf.summary.FileWriter(logs_path + 'PC_g5g0_lr1_1e11e1_l15fc64b64m5k5kv50p50k/seed' + str(seed), sess.graph)
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

a1_goal0 = [0]
a1_goal1 = [0]
a1_goal2 = [0]
a2_goal0 = [0]
a2_goal1 = [0]
a2_goal2 = [0]
a1_g0_max_success_rate_100 = 0
a2_g0_max_success_rate_100 = 0
a1_g1_max_success_rate_100 = 0
a2_g1_max_success_rate_100 = 0
a1_g2_max_success_rate_100 = 0
a2_g2_max_success_rate_100 = 0

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

    reached1 = True
    reached2 = True

    s1 = s
    s2 = s
    m1 = 0
    m2 = 0
    while True:
        # if episode >= MAX_EPISODE:
        #     render(render=env.render_env(), pause=0.01, is_block=False)
        if reached1:
            g1 = meta_dqn1.choose_action(np.array(s1 + [m1]))
            dur1 = 0
            af1 = 0
            # execute primitive actions
            if g1 == 3 or g1 == 4:
                a1 = g1 + 2
                dur1 = 1
            else:
                # TODO
                ss = s1
                mm1 = m1
                gg1 = env.encodeGoal(g1, 0)
                ssg1 = ss + gg1

                a1 = dqn1.choose_action(np.array(ssg1 + [mm1]))
                dur1 += 1
        else:
            # TODO
            gg1 = env.encodeGoal(g1, 0)
            ssg1 = ss + gg1

            a1 = dqn1.choose_action(np.array(ssg1 + [mm1]))
            dur1 += 1

        if reached2:
            g2 = meta_dqn2.choose_action(np.array(s2 + [m2]))
            dur2 = 0
            af2 = 0
            # execute primitive actions
            if g2 == 3 or g2 == 4:
                a2 = g2 + 2
                dur2 = 1
            else:
                # TODO
                ss = s2
                mm2 = m2
                gg2 = env.encodeGoal(g2, 1)
                ssg2 = ss + gg2

                a2 = dqn2.choose_action(np.array(ssg2 + [mm2]))
                dur2 += 1
        else:
            # TODO
            gg2 = env.encodeGoal(g2, 1)
            ssg2 = ss + gg2

            a2 = dqn2.choose_action(np.array(ssg2 + [mm2]))
            dur2 += 1

        ss_, mm1_, mm2_, ir1, ir2, f1, f2, done, reached1, reached2 = env.step(a1, a2, g1, g2)

        af1 += f1
        af2 += f2

        # store transition for goal steps
        if g1 in range(3):
            ss_g1 = ss_ + gg1
            dqn1.store_transition(ssg1 + [mm1], a1, ir1, ss_g1 + [mm1_])

            if reached1 or dur1 >= MAX_GOALSTEP:
                s1_ = ss_
                m1_ = mm1_
                if reached1:
                    if g1 == 0:
                        a1_goal0.append(1)
                    elif g1 == 1:
                        a1_goal1.append(1)
                    elif g1 == 2:
                        a1_goal2.append(1)
                else:
                    if g1 == 0:
                        a1_goal0.append(0)
                    elif g1 == 1:
                        a1_goal1.append(0)
                    elif g1 == 2:
                        a1_goal2.append(0)

                meta_dqn1.store_transition(s1 + [m1], g1, af1, dur1, s1_ + [m1_])
                s1 = s1_
                m1 = m1_
                # TODO
                reached1 = True
        else:
            s1_ = ss_
            m1_ = mm1_
            meta_dqn1.store_transition(s1 + [m1], g1, af1, dur1, s1_ + [m1_])
            reached1 = True
            s1 = s1_
            m1 = m1_

        if g2 in range(3):
            ss_g2 = ss_ + gg2
            dqn2.store_transition(ssg2 + [mm2], a2, ir2, ss_g2 + [mm2_])

            if reached2 or dur2 >= MAX_GOALSTEP:
                s2_ = ss_
                m2_ = mm2_
                if reached2:
                    if g2 == 0:
                        a2_goal0.append(1)
                    elif g2 == 1:
                        a2_goal1.append(1)
                    elif g2 == 2:
                        a2_goal2.append(1)
                else:
                    if g2 == 0:
                        a2_goal0.append(0)
                    elif g2 == 1:
                        a2_goal1.append(0)
                    elif g2 == 2:
                        a2_goal2.append(0)

                meta_dqn2.store_transition(s2 + [m2], g2, af2, dur2, s2_ + [m2_])
                s2 = s2_
                m2 = m2_
                # TODO
                reached2 = True
        else:
            s2_ = ss_
            m2_ = mm2_
            meta_dqn2.store_transition(s2 + [m2], g2, af2, dur2, s2_ + [m2_])
            reached2 = True
            s2 = s2_
            m2 = m2_

        ar1 += f1
        ar2 += f2

        episode_step += 1
        total_step += 1

        # train networks
        if (total_step > before_learning) and (total_step % train_interval == 0):
            if not stop_flag:
                train_time += 1
                dqn1.learn()
                dqn2.learn()

            if IS_STOP \
                    and (a1_g0_success_rate_recent_100 > 0.85 and a2_g0_success_rate_recent_100 > 0.85
                    and a1_g1_success_rate_recent_100 > 0.85 and a2_g1_success_rate_recent_100 > 0.85
                    and a1_g2_success_rate_recent_100 > 0.85 and a2_g2_success_rate_recent_100 > 0.85):
                stop_flag = True

            if train_time > pre_train:
                meta_dqn1.learn()
                meta_dqn2.learn()

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
        ss = ss_

    a1_goal0 = a1_goal0[-100:]
    a1_goal1 = a1_goal1[-100:]
    a1_goal2 = a1_goal2[-100:]
    a2_goal0 = a2_goal0[-100:]
    a2_goal1 = a2_goal1[-100:]
    a2_goal2 = a2_goal2[-100:]
    a1_g0_success_rate_recent_100 = sum(a1_goal0) / 100.0
    a1_g1_success_rate_recent_100 = sum(a1_goal1) / 100.0
    a1_g2_success_rate_recent_100 = sum(a1_goal2) / 100.0
    a2_g0_success_rate_recent_100 = sum(a2_goal0) / 100.0
    a2_g1_success_rate_recent_100 = sum(a2_goal1) / 100.0
    a2_g2_success_rate_recent_100 = sum(a2_goal2) / 100.0

    a1_g0_max_success_rate_100 = a1_g0_max_success_rate_100 if a1_g0_success_rate_recent_100 < a1_g0_max_success_rate_100 else a1_g0_success_rate_recent_100
    a1_g1_max_success_rate_100 = a1_g1_max_success_rate_100 if a1_g1_success_rate_recent_100 < a1_g1_max_success_rate_100 else a1_g1_success_rate_recent_100
    a1_g2_max_success_rate_100 = a1_g2_max_success_rate_100 if a1_g2_success_rate_recent_100 < a1_g2_max_success_rate_100 else a1_g2_success_rate_recent_100
    a2_g0_max_success_rate_100 = a2_g0_max_success_rate_100 if a2_g0_success_rate_recent_100 < a2_g0_max_success_rate_100 else a2_g0_success_rate_recent_100
    a2_g1_max_success_rate_100 = a2_g1_max_success_rate_100 if a2_g1_success_rate_recent_100 < a2_g1_max_success_rate_100 else a2_g1_success_rate_recent_100
    a2_g2_max_success_rate_100 = a2_g2_max_success_rate_100 if a2_g2_success_rate_recent_100 < a2_g2_max_success_rate_100 else a2_g2_success_rate_recent_100

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
              '. g0_sr:', a1_g0_success_rate_recent_100, '(', a1_g0_max_success_rate_100, ')', '/',
              a2_g0_success_rate_recent_100, '(', a2_g0_max_success_rate_100, ')',
              '. g1_sr:', a1_g1_success_rate_recent_100, '(', a1_g1_max_success_rate_100, ')', '/',
              a2_g1_success_rate_recent_100, '(', a2_g1_max_success_rate_100, ')',
              '. g2_sr:', a1_g2_success_rate_recent_100, '(', a1_g2_max_success_rate_100, ')', '/',
              a2_g2_success_rate_recent_100, '(', a2_g2_max_success_rate_100, ')',
              '. Succ_rate:', success_rate_recent_100, '/', max_success_rate_100, ',',
              success_rate_recent_1000, '/', max_success_rate_1000,
              '. Accm_r: %.2f' % a1_rs_100, '(%.2f)' % a1_rs_max_100, '/ %.2f' % a2_rs_100, '(%.2f)' % a2_rs_max_100,
              '. Steps: %.2f' % game_steps_100)


print('---')
print('END')

