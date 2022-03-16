import tensorflow as tf
from algorithm.common.tf_utils import function as Function
# ================================================================
# Reinforcement Learning related
# ================================================================

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return Function([], [], updates=[expression])

import math
def natural_exp_inc(init_param, max_param, global_step, current_step, inc_step=1000, inc_rate=0.5, stair_case=False):
    '''
    :param init_param:  初始参数大小
    :param max_param:  最大参数（参数增长上界）
    :param global_step: 全局增长步数
    :param current_step: 当前步数
    :param inc_step: 实现增长的频率
    :param inc_rate: 增长率
    :param stair_case: 是否阶梯状增长
    :return:
    '''
    p = (global_step - current_step) / inc_step
    if stair_case:
        p = math.floor(p)
    increased_param = min((max_param - init_param) * math.exp(-inc_rate * p) + init_param, max_param)
    return increased_param


def natural_exp_decay(init_param, min_param, global_step, current_step, decay_step, decay_rate=0.5, stair_case=False):
    '''
    :param init_param: 初始参数大小
    :param min_param: 最小参数
    :param global_step: 全局下降步数
    :param current_step: 当前步数
    :param decay_step: 实现衰减的频率
    :param decay_rate: 衰减率
    :param stair_case: 是否阶梯状衰减
    :return:
    '''
    p = current_step / decay_step
    if stair_case:
        p = math.floor(p)
    decayed_param = max((init_param - min_param) * math.exp(-decay_rate * p) + min_param, min_param)
    return decayed_param