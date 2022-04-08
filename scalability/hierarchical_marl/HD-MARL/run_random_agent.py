#!/usr/bin/env python3
# encoding=utf-8

import os
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../../')



class RandomAgent(object):
    def __init__(self, action_num):
        self.action_num = action_num

    def choose_action(self, state):
        action = np.random.choice(range(self.action_num))
        return action

def render(render, pause, is_block=False):
    plt.imshow(render)
    plt.show(block=is_block)
    plt.pause(pause)
    plt.clf()

def args_parser():
    # --------------------------------- Param parser ------------------------------
    parser = argparse.ArgumentParser(description='manual to this script')
    # parser.add_argument('--env', type=str, default='Room')
    # parser.add_argument('--env', type=str, default='Ring')
    parser.add_argument('--env', type=str, default='Coordination')
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--max-timestep', type=int, default=100)
    # parser.add_argument('--is-render', type=int, default=0)
    parser.add_argument('--is-render', type=int, default=1)

    args = parser.parse_args()
    print("-------------------------------------------------")
    print("Params:")
    print("--env:", args.env)
    print("--seed:", args.seed)
    print("--max-timestep:", args.max_timestep)
    print("-------------------------------------------------")

    return args

def run(args):

    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    MAX_EPISODE = 100000
    MAX_TIMESTEP = args.max_timestep

    is_fixed = True
    action_num = 7

    # - init env
    if args.env == 'Room':
        from envs.MATC_2a2d2t_Room.MATC_Room_non_hie_with_common_reward import GameEnv
        env = GameEnv(is_fixed=is_fixed)
    elif args.env == 'Ring':
        from envs.MATC_2a2d8t_Ring.MATC_Ring_non_hie_common_reward import GameEnv
        env = GameEnv(is_fixed=is_fixed)
    elif args.env == 'Coordination':
        from envs.classic_coordination_games.MATC_PC_non_hie import GameEnv
        env = GameEnv()
    else:
        raise NotImplementedError

    # - init agents, e.g., random agents
    agent1 = RandomAgent(action_num=action_num)
    agent2 = RandomAgent(action_num=action_num)


    # - init
    episode = 0
    success_count = 0
    total_step = 0
    result_list = [0]

    max_success_rate_100 = 0
    max_success_rate_1000 = 0

    print_interval = 500

    a1_rs = [0]
    a2_rs = [0]
    a1_rs_max_100 = -1
    a2_rs_max_100 = -1
    game_steps = [0]


    # --------------------------------- training ------------------------------

    while episode < MAX_EPISODE:
        s = env.reset()
        # - m1 and m2 denotes whether the agent is carrying something
        m1, m2 = 0, 0

        ar1, ar2 = 0, 0
        episode_step = 0

        while True:
            if args.is_render:
                render(render=env.render_env(), pause=0.01, is_block=False)

            a1 = agent1.choose_action(np.array(s + [m1]))
            a2 = agent2.choose_action(np.array(s + [m2]))

            s_, m1_, m2_, r1, r2, done = env.step(a1, a2)

            ar1 += r1
            ar2 += r2

            episode_step += 1
            total_step += 1

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

        if episode % print_interval == 0:
            print('--Episode:', episode,
                  '. Succ_rate:', success_rate_recent_100, '/', max_success_rate_100, ',',
                  success_rate_recent_1000, '/', max_success_rate_1000,
                  '. Accm_r: %.2f' % a1_rs_100, '(%.2f)' % a1_rs_max_100, '/ %.2f' % a2_rs_100, '(%.2f)' % a2_rs_max_100,
                  '. Steps: %.2f' % game_steps_100)


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    run(args)

    print('---')
    print('Time elapsed:', time.time() - start_time)
    print('---')

