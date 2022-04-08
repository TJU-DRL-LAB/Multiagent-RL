import argparse
import random
import json
import numpy as np

import time

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

from neural.a2c_trainer import RLStatistics

from tensorboardX import SummaryWriter

import os

try:
    import thread
except ImportError:
    import _thread as thread

import multiprocessing
import multiprocessing.connection
import pickle
import numpy as np

from torch import cuda

def execute_runner(runner, args, addr):
    runner(args, addr).run()

class MultiRunner:
    def __init__(self, args, addr):
        self.init_trainer(args)
        self.addr = self.get_real_addr(addr)
        self.conn = multiprocessing.connection.Client(self.addr)

    def init_trainer(self, args):
        if args.gpuid:
            print('Running with GPU {}.'.format(args.gpuid[0]))
            cuda.set_device(args.gpuid[0])
        else:
            print('Running with CPU.')

        if args.random_seed:
            random.seed(args.random_seed+os.getpid())
            np.random.seed(args.random_seed+os.getpid())

        schema = Schema(args.schema_path)
        scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
        valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)

        # if len(args.agent_checkpoints) == 0
        # assert len(args.agent_checkpoints) <= len(args.agents)
        if len(args.agent_checkpoints) < len(args.agents):
            ckpt = [None] * 2
        else:
            ckpt = args.agent_checkpoints

        systems = [get_system(name, args, schema, False, ckpt[i], id=i) for i, name in enumerate(args.agents)]

        rl_agent = 0
        system = systems[rl_agent]
        model = system.env.model
        loss = None
        # optim = build_optim(args, [model, system.env.critic], None)
        optim = {'model': build_optim(args, model, None),
                 'critic': build_optim(args, system.env.critic, None)}
        optim['critic']._set_rate(0.05)

        scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
        from neural.a2c_trainer import RLTrainer as A2CTrainer
        trainer = A2CTrainer(systems, scenarios, loss, optim, rl_agent,
                             reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)

        self.args = args
        self.trainer = trainer
        self.systems = systems

    def get_real_addr(self, addr):
        return addr

    def simulate(self, cmd):
        raise NotImplementedError

    def train(self, cmd):
        raise NotImplementedError

    def update_model(self, cmd):
        raise NotImplementedError

    def fetch_model(self, cmd):
        raise NotImplementedError

    def valid(self, cmd):
        raise NotImplementedError

    def save_model(self, cmd):
        raise NotImplementedError

    def run(self):
        while True:
            cmd = self.conn.recv()

            print('recv: ', cmd[0])
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'check':
                self.conn.send(['done'])
            elif cmd[0] == 'simulate':
                data = self.simulate(cmd[1:])
                self.conn.send(['done', pickle.dumps(data)])
                # try:
                # except Exception, err:
                #     print(e)
                #     self.conn.send(['error'])
            elif cmd[0] == 'train':
                data = self.train(pickle.loads(cmd[1]))
                self.conn.send(['done', pickle.dumps(data)])
                # try:
                #     data = self.train(pickle.loads(cmd[1]))
                #     self.conn.send(['done', pickle.dumps(data)])
                # except:
                #     self.conn.send(['error'])
            elif cmd[0] == 'update_model':
                self.update_model((cmd[1],) + pickle.loads(cmd[2]))
                self.conn.send(['done'])
                # try:
                #     self.update_model(pickle.loads(cmd[1]))
                #     self.conn.send(['done'])
                # except:
                #     self.conn.send(['error'])

            elif cmd[0] == 'fetch_model':

                data = self.fetch_model(cmd[1:])
                self.conn.send(['done', pickle.dumps(data)])
                # try:
                #     data = self.fetch_model(cmd[1:])
                #     self.conn.send(['done', pickle.dumps(data)])
                # except:
                #     self.conn.send(['error'])
            elif cmd[0] == 'valid':
                data = self.valid(cmd[1])
                self.conn.send(['done', pickle.dumps(data)])

            elif cmd[0] == 'save_model':
                self.save_model(pickle.loads(cmd[1]))
                self.conn.send(['done'])
