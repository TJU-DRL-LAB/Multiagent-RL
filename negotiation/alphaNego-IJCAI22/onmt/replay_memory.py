import numpy as np
import scipy.signal
# from mpi_tools import mpi_statistics_scalar
import torch

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=20000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def clear(self):
        self.storage = []
        self.ptr = 0

    def add(self, data):

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_uttr, batch_state, batch_intent_act, batch_price_act, batch_price_mask, batch_next_uttr, batch_next_state, batch_reward, batch_done = [], [], [], [], [], [], [], [], []

        for i in ind:
            state_0, state_1, uttr, intent_act, price_act, price_mask, next_state_0, next_state1, next_uttr, reward, done = self.storage[i]

            state = torch.cat((state_0, state_1), dim=0)
            next_state = torch.cat((next_state_0, next_state1), dim=0)

            batch_uttr.append(uttr)
            batch_state.append(state)
            batch_intent_act.append(intent_act)
            batch_price_act.append(price_act)
            batch_price_mask.append(price_mask)

            batch_next_uttr.append(next_uttr)
            batch_next_state.append(next_state)
            batch_reward.append(reward)
            batch_done.append(done)

        return batch_uttr, batch_state, batch_intent_act, batch_price_act, batch_price_mask, batch_next_uttr, batch_next_state, batch_reward, batch_done

