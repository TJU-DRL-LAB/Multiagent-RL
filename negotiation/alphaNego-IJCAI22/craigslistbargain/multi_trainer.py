import argparse
import random
import json
import numpy as np

from onmt.Utils import use_gpu

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
import cocoa.options

from core.scenario import Scenario
from core.controller import Controller
from systems import get_system
from neural.rl_trainer import RLTrainer
from neural import build_optim
import options

import torch

from multi_manager import MultiRunner

class MultiTrainer(MultiRunner):
    def __init__(self, args, addr):
        super(MultiTrainer, self).__init__(args, addr)

    def simulate(self, cmd):
        i, batch_size, real_batch = cmd
        data = self.trainer.sample_data(i, batch_size, self.args, real_batch=real_batch)
        return data

    def train(self, cmd):
        epoch, batches, rewards, train_mode = cmd
        if train_mode == 'normal':
            pretrain_number = 3
            for i in range(pretrain_number):
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, fix_policy=True)

            info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                           discount=self.args.discount_factor)

            for i in range(pretrain_number):
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, fix_policy=True)
        elif train_mode == 'fix_value':
            info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                           discount=self.args.discount_factor, fix_value=True)
        elif train_mode == 'fix_policy':
            info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                           discount=self.args.discount_factor, fix_policy=True)
        else:
            info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                           discount=self.args.discount_factor)
        return info

    def valid(self, cmd):
        if len(cmd) == 2:
            start, length = cmd
            infos = self.trainer.validate(self.args, length, start=start)
        else:
            start, length, split, exchange = cmd
            infos = self.trainer.validate(self.args, length, start=start, split=split, exchange=exchange)

        return infos

    def save_model(self, cmd):
        i, valid_stats = cmd
        self.trainer.drop_checkpoint(self.args, i + 1, valid_stats,
                                     model_opt=self.trainer.agents[self.trainer.training_agent].env.model_args)
        # if self.args.update_oppo:
        #     self.trainer.update_opponent(['policy', 'critic'])

    def update_model(self, cmd):
        model_idx, model_p, critic_p = cmd
        env = self.systems[model_idx].env

        env.model.load_state_dict(model_p)
        env.critic.load_state_dict(critic_p)

    def fetch_model(self, cmd):
        model_idx = cmd[0]
        env = self.systems[model_idx].env
        return env.model.state_dict(), env.critic.state_dict()


