import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns


class PolicyCounter(object):
    def __init__(self, num, from_dataset=True, only_incorrect=False):
        self.from_dataset = from_dataset
        self.only_incorrect = only_incorrect
        self.counter = np.zeros((num, num))
        self.num = np.zeros((num,))
        self.policy = np.zeros((num, num))

        self.num_ds = {}
        self.num_broken_ds = {}

        self.type = None
        if from_dataset:
            self.type = 'dataset'
        elif only_incorrect:
            self.type = 'incorrect'
        else:
            self.type = 'model'

    def add_one(self, state, act):
        self.counter[state, act] += 1

    def add_dialogue(self, d):
        d.lf_to_int()
        n = len(d.lfs)
        for i in range(n-1):
            self.add_one(d.lfs[i]['intent'], d.lfs[i+1]['intent'])

    def add_dialogues(self, ds):
        if isinstance(ds, dict):
            for k in ds:
                self.add_dialogues(ds[k])
            return
        for i, d in enumerate(ds):
            self.add_dialogue(d)

    @staticmethod
    def _get_softmax(policy):
        policy = np.exp(policy - np.max(policy, axis=1, keepdims=True))
        policy = policy / np.sum(policy, axis=1, keepdims=True)
        return policy

    def update_from_batch(self, batch, policy):
        return
        n = batch.size
        state = batch.encoder_intent.cpu().numpy()
        policy = policy.data.cpu().numpy()
        policy = self._get_softmax(policy)
        if self.type is 'model':
            for i in range(n):
                self.update_policy(state[i,:], policy[i,:])
        elif self.type is 'incorrect':
            label = batch.target_intent.cpu().numpy()
            for i in range(n):
                if label[i] == np.argmax(policy[i,:]) or label[i] == 19:
                    continue
                tmp = np.zeros_like(policy[i,:])
                tmp[label[i]] = 1
                self.update_policy(state[i,:], tmp)

    def update_policy(self, state, policy):
        self.num[state] += 1
        self.counter[state, :] += policy

    def get_policy(self):
        if self.type is 'model':
            tmp = self.num.reshape(-1,1).copy()
            tmp[tmp<1] = 1
            # print('counter: ', self.counter)
            # print('num: ', tmp)
            self.policy = self.counter / tmp
        elif self.type is 'incorrect':
            tmp = self.num.reshape(-1, 1).copy()
            tmp = np.sum(tmp)
            # tmp[tmp < 1] = 1
            # print('counter: ', self.counter)
            # print('num: ', tmp)
            self.policy = self.counter / tmp
        elif self.type is 'dataset':
            tmp = np.sum(self.counter, axis=1, keepdims=True)
            tmp[tmp<1] = 1
            # print('counter: ', self.counter)
            # print('num: ', tmp)
            self.policy = self.counter / tmp
        return self.policy

    def draw(self, ax=None):
        self.get_policy()
        # print(self.policy)
        sns.heatmap(self.policy, annot=False, ax=ax)

    @classmethod
    def draw_policy(cls, policies=[], filename=None):
        n = len(policies)
        f, ax = plt.subplots(nrows=n)
        print('drawing: {}'.format([i.type for i in policies]))
        for i in range(n):
            policies[i].draw(ax[i])

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)