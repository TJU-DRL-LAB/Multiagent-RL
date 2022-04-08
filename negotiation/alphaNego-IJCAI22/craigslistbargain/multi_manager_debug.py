import argparse
import copy
import random
import json
import numpy as np
import json

import time

from onmt.Utils import use_gpu
import logging

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
from buffer import ReplayBuffer
import torch

try:
    import thread
except ImportError:
    import _thread as thread

import multiprocessing
import multiprocessing.connection
import math
import pickle as pkl
import numpy as np
import shutil
from neural.dsac_utils import *
from multi_manager import MultiRunner, execute_runner

def init_dir(path, clean_all=False):
    if not os.path.exists(path):
        print('[Info] make dir {}'.format(path))
        os.mkdir(path)
    else:
        print('[Warning] path {} exists!'.format(path))
        if clean_all:
            print('[Warning] clean files in {}!'.format(path))
            shutil.rmtree(path, True)
            # Deal with delay on NAS
            while not os.path.exists(path):
                os.mkdir(path)
            print('[Info] remake dir {}'.format(path))


class MultiRunner:
    def __init__(self, args, addr):
        self.init_trainer(args)
        # self.addr = self.get_real_addr(addr)
        # self.conn = multiprocessing.connection.Client(self.addr)

    def init_trainer(self, args):
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
        if args.self_play:
            systems.append(get_system("pt-neural-dsac", args, schema, False, ckpt[0], id=2))
        rl_agent = 0
        system = systems[rl_agent]
        model = None
        optim = None
        loss = None
        scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
        if system.env.name == 'pt-neural-dsac':
            model = system.env.dsac
            from neural.dsac_trainer import DSACTrainer as DSACTrainer
            dsac_trainer = DSACTrainer(systems, scenarios, loss, optim, rl_agent,
                                       reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)
            self.trainer = dsac_trainer
        else:
            model = system.env.model

            # optim = build_optim(args, [model, system.env.critic], None)
            optim = {'model': build_optim(args, model, None),}
            if system.env.critic is not None:
                optim['critic'] = build_optim(args, system.env.critic, None)
                optim['critic']._set_rate(0.05)
            if system.env.tom_model is not None:
                optim['tom'] = build_optim(args, system.env.tom_model, None)
                if args.tom_model not in ['naive', 'history']:
                    optim['tom_identity'] = build_optim(args, system.env.tom_model.encoder.identity, None)
                # optim['tom']._set_rate(0.1)
            from neural.a2c_trainer import RLTrainer as A2CTrainer
            trainer = A2CTrainer(systems, scenarios, loss, optim, rl_agent,
                                 reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)

            self.trainer = trainer




        self.args = args


        self.systems = systems

    def get_real_addr(self, addr):
        return addr

    def simulate(self, cmd):
        i, batch_size, real_batch = cmd
        data = self.trainer.sample_data(i, batch_size, self.args, real_batch=real_batch)
        return data

    def simulate_pfsp(self, cmd):
        i, batch_size, real_batch = cmd
        data = self.trainer.sample_data_pfsp(i, batch_size, self.args, real_batch=real_batch)
        return data

    def train(self, epoch, batches, rewards, train_mode):
        update_table = {'policy': True, 'value': True}
        with torch.autograd.set_detect_anomaly(True):
            if train_mode == 'normal':
                pretrain_number = 3
                update_table['policy'] = False
                for i in range(pretrain_number):
                    info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                                   discount=self.args.discount_factor, update_table=update_table)
                update_table['policy'] = True
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
                update_table['policy'] = False
                for i in range(pretrain_number):
                    info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                                   discount=self.args.discount_factor, update_table=update_table)
            elif train_mode == 'fix_value':
                update_table['value'] = False
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
            elif train_mode == 'fix_policy':
                update_table['policy'] = False
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
            else:
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
        return info
  
    def trainset_valid(self, i, batch_size, real_batch, ):
        data = self.trainer.sample_data(i, batch_size, self.args, real_batch=real_batch, eval=True)
        return data

    def get_eval_dict(self, examples, strategies):
        ret = self.trainer.get_eval_dict(examples, strategies)
        return ret

    def get_eval_dict_sp(self, examples, strategies):
        ret = self.trainer.get_eval_dict_sp(examples, strategies)
        return ret

    def valid(self, start, length):
        infos = self.trainer.validate(self.args, length, start=start) #  [total_stats, oppo_total_stats], examples, verbose_str
        return infos

    def valid_sp(self, start, length, opponent):
        infos = self.trainer.validate_sp(self.args, length, start=start, opponent=opponent)
        return infos

    def save_model(self, i, valid_stats, score_type):
        self.trainer.drop_checkpoint(self.args, i + 1, valid_stats,
                                     model_opt=self.trainer.agents[self.trainer.training_agent].env.model_args,
                                     score_type=score_type)
        # if self.args.update_oppo:
        #     self.trainer.update_opponent(['policy', 'critic'])

    def save_best_model(self, i, valid_stats, score_type, best_only):

        self.trainer.drop_checkpoint(self.args, i + 1, valid_stats,
                                     model_opt=self.trainer.agents[self.trainer.training_agent].env.model_args,
                                     score_type=score_type, best_only=best_only)

    def update_model(self, cmd):
        model_idx, model_p, critic_p = cmd
        env = self.systems[model_idx].env

        env.model.load_state_dict(model_p)
        env.critic.load_state_dict(critic_p)

    def fetch_model(self, cmd):
        model_idx = cmd[0]
        env = self.systems[model_idx].env
        return env.model.state_dict(), env.critic.state_dict()

    def train_tom(self, model_idx, batch_iters, strategy, update_table=None, ret_table=None, dump_file=None):
        env = self.systems[model_idx].env
        # if learn_type == 'id':
        #     update = {'id': True, 'tom': False}
        #     ret = {'id': True, 'tom': False}
        # elif learn_type == 'id_tom':
        #     update = {'id': True, 'tom': True}
        #     ret = {'id': True, 'tom': True}
        # elif learn_type == 'fixed_id_tom':
        #     update = {'id': False, 'tom': True}
        #     ret = {'id': True, 'tom': True}
        # elif learn_type in ['history', 'naive']:
        #     update = {'id': False, 'tom': True}
        #     ret = {'id': False, 'tom': True}
        # else:
        #     raise NameError('unknown learn_type ')

        train_loss = self.trainer.update_tom(self.args, batch_iters, strategy, env.tom_model,
                                             update_table=update_table,
                                             ret_table=ret_table, dump_name=dump_file)
        return train_loss

    def valid_tom(self, model_idx, batch_iters, strategy, update_table=None, ret_table=None, dump_file=None):
        env = self.systems[model_idx].env
        update_table = {'id': False, 'tom': False}
        valid_loss = self.trainer.update_tom(self.args, batch_iters, strategy,
                                             env.tom_model, update_table=update_table,
                                             ret_table=ret_table, dump_name=dump_file)
        return valid_loss

    def split_batch(self, batch_iters, batch_size, device=None):
        ret = self.trainer._sort_merge_batch(batch_iters, batch_size, device=device)
        return ret

    def add_strategy_in_language(self, batch_iters, strategies):
        self.trainer.add_strategy_in_language(batch_iters, strategies)

    def send(self, cmd):
        if cmd[0] == 'quit':
            return
        elif cmd[0] == 'check':
            # self.conn.send(['done'])
            return ['done']
        elif cmd[0] == 'simulate':
            data = self.simulate(cmd[1:])
            return ['done', pkl.dumps(data)]

        elif cmd[0] == 'simulate_pfsp':
            data = self.simulate_pfsp(cmd[1:])
            return ['done', pkl.dumps(data)]

        elif cmd[0] == 'train':
            data = self.train(pkl.loads(cmd[1]))
            return ['done', pkl.dumps(data)]

        elif cmd[0] == 'update_model':
            self.update_model((cmd[1],) + pkl.loads(cmd[2]))
            return ['done']


        elif cmd[0] == 'fetch_model':

            data = self.fetch_model(cmd[1:])
            return ['done', pkl.dumps(data)]

        elif cmd[0] == 'valid':
            data = self.valid(cmd[1])
            return ['done', pkl.dumps(data)]
            # self.conn.send(['done', pkl.dumps(data)])

        elif cmd[0] == 'save_model':
            self.save_model(pkl.loads(cmd[1]))
            return ['done']
            # self.conn.send(['done'])

        else:
            # Using Universal Formation
            if len(cmd) < 2:
                cmd.append([])
            else:
                cmd[1] = pkl.loads(cmd[1])
            if len(cmd) < 3:
                cmd.append({})
            else:
                cmd[2] = pkl.loads(cmd[2])

            # try:
            ret = getattr(self, cmd[0])(*cmd[1], **cmd[2])
            status = 'done'
            ret_data = ret

            ret_data = pkl.dumps(ret_data)
            return [status, ret_data]

    def local_send(self, cmd):
        if len(cmd) < 2:
            cmd.append([])
        if len(cmd) < 3:
            cmd.append({})

        # try:
        ret = getattr(self, cmd[0])(*cmd[1], **cmd[2])
        status = 'done'
        ret_data = ret
        return [status, ret_data]

    def train_dsac(self, batches, update_policy, update_critic):
        zf1_loss = None
        zf2_loss = None
        policy_loss = None
        alpha_intent = None
        alpha_price = None
        q_loss = None
        kl_loss = None
        if update_policy is True and update_critic is True:
            print("train actor and critic")
            for i in range(10):
                zf1_loss, zf2_loss, _,_,_,_,_ = self.trainer.train_from_torch(batches, update_policy=False, update_critic=True, is_pretrain=False)
            for i in range(5):
                _, _, policy_loss, alpha_intent, alpha_price, q_loss, kl_loss = self.trainer.train_from_torch(batches, update_policy=True, update_critic=False, is_pretrain=False)
        elif update_policy is False and update_critic is True:
            print("pretrain critic")
            for i in range(10):
                zf1_loss, zf2_loss,  _,_,_,_,_ = self.trainer.train_from_torch(batches, update_policy=False, update_critic=True, is_pretrain=True)
        return zf1_loss, zf2_loss, policy_loss, alpha_intent, alpha_price,q_loss, kl_loss

class MultiManager():
    def __init__(self, num_cpu, args, worker_class):
        self.local_workers = []
        self.worker_addr = []
        self.trainer_addr = []
        self.args = args

        for i in range(num_cpu):
            addr = ('localhost', 7000 + i)
            worker = worker_class(args, addr)
            self.worker_addr.append(worker)
            # self.local_workers.append(multiprocessing.Process(target=execute_runner, args=(worker_class, args, addr)))
            self.local_workers.append(worker)
        # self.trainer = multiprocessing.Process(target=execute_runner, args=(trainer_class, args))
        self.trainer = self.local_workers[0]

        self.worker_listener = []
        for i, addr in enumerate(self.worker_addr):
            # self.worker_listener.append(multiprocessing.connection.Listener(addr))
            self.worker_listener.append(addr)
        self.worker_conn = []

        cache_path = 'cache/{}'.format(args.name)
        log_path = 'logs/' + args.name
        init_dir(cache_path)
        init_dir(log_path, clean_all=True)

        self.writer = SummaryWriter(logdir='logs/{}'.format(args.name))
        self.policies_log = [{}, {}]

    def run_local_workers(self):
        for w in self.local_workers:
            w.start()

    def update_worker_list(self):
        self.worker_conn = []
        for l in self.worker_listener:
            # self.worker_conn.append(l.accept())
            self.worker_conn.append(l)
        return len(self.worker_conn)

    @staticmethod
    def allocate_tasks(num_worker, batch_size):
        ret = []
        while num_worker > 0:
            ret.append(batch_size // num_worker)
            batch_size -= ret[-1]
            num_worker -= 1

        print('allocate: {} workers, {} tasks, final list:{}'.format(num_worker, batch_size, ret))
        return ret

    def _draw_tensorboard(self, ii, losses, all_rewards):
        # print(all_rewards)
        for j in range(2):
            self.writer.add_scalar('agent{}/reward'.format(j), np.mean(all_rewards[j]), ii)
            if len(losses[j]) > 0:
                for k in losses[j][0]:
                    tmp = []
                    for l in losses[j]:
                        tmp.append(l[k])
                    tmp = np.concatenate(tmp[j])
                    tmp = np.mean(tmp)
                    self.writer.add_scalar('agent{}/{}'.format(j, k), tmp, ii)
        self.writer.flush()

    def _draw_tensorboard_valid(self, ii, all_rewards):
        for j in range(2):
            self.writer.add_scalar('agent{}/dev_reward'.format(j), all_rewards[j], ii)

    def dump_examples(self, examples, verbose_strs, epoch, mode='train', other_path=None):
        # Dump with details
        args = self.args
        if other_path is None:
            path = args.model_path
        else:
            path = other_path
        path_txt = '{root}/{model}_{mode}_example{epoch}.txt'.format(
            root=path,
            model=args.name,
            mode=mode,
            epoch=epoch)
        path_pkl = '{root}/{model}_{mode}_example{epoch}.pkl'.format(
            root=path,
            model=args.name,
            mode=mode,
            epoch=epoch)

        print('Save examples at {} and {}.'.format(path_txt, path_pkl))
        with open(path_txt, 'w') as f:
            for ex in verbose_strs:
                f.write('-' * 7 + '\n')
                for s in ex:
                    f.write(s + '\n')
        with open(path_pkl, 'wb') as f:
            pkl.dump(examples, f)



    def evaluate(self):
        num_worker = self.update_worker_list()
        worker = self.worker_conn[0]
        args = self.args
        sample_size = args.batch_size
        max_epoch = args.epochs
        last_time = time.time()
        if args.debug:
            sample_size = 2

        eval_dict = {}
        separate_edict = [{} for _ in range(10)]

        # add dd to d
        def update_edict(d, dd):
            for k in dd:
                if d.get(k) is None:
                    d[k] = []
                d[k] = d[k] + dd[k]


        def get_result_dict(d):
            ret = {}
            if d.get('reward') is None:
                num = 0
            else:
                num = len(d.get('reward'))
            for k in d:

                ret[k] = np.mean(d[k])
                ret[k + '_std'] = np.std(d[k])
            return ret, num

        for epoch in range(max_epoch):
            last_time = time.time()

            info = worker.local_send(['trainset_valid', (epoch, sample_size, sample_size)])
            _batch_iters, batch_info, example, v_str = info[1]
            _rewards, strategies = batch_info

            data_pkl = 'cache/{}/data_{}.pkl'.format(args.name, epoch)
            with open(data_pkl, 'wb') as f:
                pkl.dump(info[1], f)

            self.dump_examples(example, v_str, epoch, other_path='logs/'+args.name)

            info = worker.local_send(['get_eval_dict', (example, strategies[1])])
            ed, sed = info[1]

            # log eval table as json file
            eval_json = 'logs/{}/eval_{}.json'.format(args.name, epoch)
            update_edict(eval_dict, ed)
            tmpd, _ = get_result_dict(eval_dict)
            tmpd['number'] = (epoch+1) * sample_size
            with open(eval_json, 'w') as f:
                json.dump(tmpd, f)

            print('=' * 5 + ' [reward: {:.3f}\t utility: {:.3f}\t success_rate: {:.3f}]'.
                  format(tmpd['reward'], tmpd['utility'], tmpd["success_rate"]))

            eval_json = 'logs/{}/eval_separate_{}.json'.format(args.name, epoch)
            tmpds = []
            for i in range(len(sed)):
                update_edict(separate_edict[i], sed[i])
                tmpd, num = get_result_dict(separate_edict[i])
                tmpd['number'] = num
                tmpd['strategy'] = i
                tmpds.append(tmpd)
            with open(eval_json, 'w') as f:
                json.dump(tmpds, f)

            print('=' * 5 + ' [Epoch {}/{}, {} dialogues for {:.3f}s.]'.
                  format(epoch + 1, max_epoch, (epoch+1)*sample_size, time.time() - last_time))


    def learn_identity(self):

        args = self.args
        save_every = 100

        batch_size = 100
        split_results = False
        if args.only_run:
            batch_size = 1

        if args.tom_model in ['id', 'uttr_id']:
            update_table = {'id': True, 'tom': False}
            ret_table = {'id': True, 'tom': False}
        elif args.tom_model in ['uttr_fid_history_tom']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['uttr_id_history_tom', 'id_tom', 'id_history_tom']:
            update_table = {'id': True, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['fixed_id_tom', 'fixed_id_history_tom']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['history', 'naive']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': False, 'tom': True}
        else:
            raise NameError('unknown learn_type ')

        if args.fix_id:
            update_table['id'] = False

        if args.only_run:
            update_table = {'id': False, 'tom': False}

        num_worker = self.update_worker_list()
        worker = self.worker_conn[0]
        train_agent = 0
        load_data = args.load_sample

        # Generate data samples or load from files
        data_pkl = 'cache/{}/data.pkl'.format(args.name)
        if load_data is None:
            print('[Info] Start sampling.')
            info = worker.send(['simulate', train_agent, args.num_dialogues, args.num_dialogues])
            with open(data_pkl, 'wb') as f:
                pkl.dump(pkl.loads(info[1]), f)
            _batch_iters, batch_info, example, v_str = pkl.loads(info[1])
        else:
            print('[Info] Load sample from {}'.format(load_data))
            info = ['done', None]
            with open(load_data, 'rb') as f:
                info[1] = pkl.load(f)
            _batch_iters, batch_info, example, v_str = info[1]

        _rewards, strategies = batch_info

        # Single Thread!
        if args.strategy_in_words:
            worker.local_send(
                ['add_strategy_in_language', (_batch_iters, strategies)]
            )
        self.dump_examples(example, v_str, 0)

        # Divide the training set
        train_size = round(len(_batch_iters[1-train_agent]) * 0.6)
        train_batch = _batch_iters[1-train_agent][:train_size]
        train_strategy = strategies[1-train_agent][:train_size]
        dev_batch = _batch_iters[1-train_agent][train_size:]
        dev_strategy = strategies[1-train_agent][train_size:]

        # if not, only learn identifier
        if args.tom_model != 'id' and split_results:
            dev_batches = [[], []]
            dev_strategies = [[], []]
            for i, s in enumerate(dev_strategy):
                dev_batches[s].append(dev_batch[i])
                dev_strategies[s].append(s)
            dev_batch = dev_batches
            dev_strategy = dev_strategies
            dev_writer = [SummaryWriter(logdir='logs/{}/strategy_{}'.format(args.name, i)) for i in range(2)]

        print('[Info] Start training model.')
        step_range = 10
        step_writer = [SummaryWriter(logdir='logs/{}/step_{}'.format(args.name, i)) for i in range(step_range)]

        # split training batch
        _, train_batch_splited = worker.local_send(
            ['split_batch', (train_batch, 1024)])
        if args.tom_model != 'id' and split_results:
            dev_batch_splited = [None, None]
            _, dev_batch_splited[0] = worker.local_send(
                ['split_batch', (dev_batch[0], 1024)]
            )
            _, dev_batch_splited[1] = worker.local_send(
                ['split_batch', (dev_batch[1], 1024)]
            )
        else:
            _, dev_batch_splited = worker.local_send(
                ['split_batch', (dev_batch, 1024)]
            )

        def draw_dev_info(loss, accu, step_info, name, w, i):
            if ret_table['id']:
                w.add_scalar('identity{}/{}_loss'.format(train_agent, name), loss[0], i)
                w.add_scalar('identity{}/{}_accuracy'.format(train_agent, name), accu[0], i)
                w.add_scalar('identity{}/{}_accuracy2'.format(train_agent, name), accu[2], i)
            if ret_table['tom']:
                w.add_scalar('tom{}/{}_intent_loss'.format(train_agent, name), loss[1], i)
                w.add_scalar('tom{}/{}_intent_accuracy'.format(train_agent, name), accu[1], i)
                w.add_scalar('tom{}/{}_price_loss'.format(train_agent, name), loss[2], i)
                w.add_scalar('tom{}/{}_total_loss'.format(train_agent, name), loss[1] + loss[2], i)
            w.flush()

        # Draw outputs on the tensorboard
        def draw_info(loss, accu, step_info, name, i):
            draw_dev_info(loss, accu, None, name, self.writer, i)

            for j, w in enumerate(step_writer):
                if j >= len(step_info[2]):
                    break
                if math.isnan(step_info[2][j]) or step_info[2][j] == 0:
                    continue
                if ret_table['id']:
                    w.add_scalar('identity{}/{}_loss'.format(train_agent, name), step_info[0][0][j], i)
                    w.add_scalar('identity{}/{}_accuracy'.format(train_agent, name), step_info[1][0][j], i)
                    w.add_scalar('identity{}/{}_accuracy2'.format(train_agent, name), step_info[1][2][j], i)

                if ret_table['tom']:
                    w.add_scalar('tom{}/{}_intent_loss'.format(train_agent, name), step_info[0][1][j], i)
                    w.add_scalar('tom{}/{}_intent_accuracy'.format(train_agent, name), step_info[1][1][j], i)

                    w.add_scalar('tom{}/{}_price_loss'.format(train_agent, name), step_info[0][2][j], i)
                w.flush()

        # train model
        cur_t = time.time()
        for i in range(args.epochs):
            # print('train.send:')
            info = worker.local_send(
                ['train_tom', (train_agent, train_batch_splited,
                               train_strategy, update_table, ret_table,
                               'cache/{}/train_pred_{}.pkl'.format(args.name, i))])
            train_loss, train_accu, train_step_info = info[1]

            if args.only_run:
                save_dir = 'logs/{}/hidden_vec_{}.pkl'.format(args.name, i)
                total_num = 0
                for j in range(len(worker.trainer.hidden_vec)):
                    assert worker.trainer.hidden_vec[j].shape[0] == worker.trainer.hidden_stra[j].shape[0], \
                        "miss match at {}, {} of {}".format(worker.trainer.hidden_vec[j].shape, worker.trainer.hidden_stra[j].shape, j)
                    total_num = total_num + len(worker.trainer.hidden_stra[j])

                with open(save_dir, "wb") as f:
                    pkl.dump([worker.trainer.hidden_vec, worker.trainer.hidden_stra], f)

                print("accu:", train_accu)
                print('[run{}/{}]\t num:{} \t time:{:.2f}s.'.format(i+1, args.epochs, total_num, time.time()-cur_t))
                continue

            draw_info(train_loss, train_accu, train_step_info, 'train', i)

            if args.tom_model != 'id' and split_results:
                # divide by 2 different id
                dev_loss = [0]*3
                dev_accu = [0]*2
                for j in range(2):
                    ratio = len(dev_strategy[j]) / (len(dev_strategy[0]) + len(dev_strategy[1]))
                    info = worker.local_send(
                        ['valid_tom', (train_agent, dev_batch_splited[j],
                                       dev_strategy[j], update_table, ret_table,
                                       'cache/{}/dev{}_pred_{}.pkl'.format(args.name, j, i))])
                    tmp_loss, tmp_accu, dev_step_info = info[1]
                    for x in range(3):
                        if isinstance(tmp_loss[x], float):
                            dev_loss[x] += ratio * tmp_loss[x]
                        else:
                            if tmp_loss[x] != [] and tmp_loss[x] is not None:
                                print(tmp_loss[x])
                            dev_loss[x] = None
                    for x in range(2):
                        if isinstance(tmp_accu[x], float):
                            dev_accu[x] += ratio * tmp_accu[x]
                        else:
                            if tmp_accu[x] != [] and tmp_accu[x] is not None:
                                print(tmp_accu[x])
                            dev_loss[x] = None
                    draw_dev_info(tmp_loss, tmp_accu, dev_step_info, 'dev', dev_writer[j], i)
                draw_dev_info(dev_loss, dev_accu, None, 'dev', self.writer, i)
            else:
                info = worker.local_send(
                    ['valid_tom', (train_agent, dev_batch_splited,
                                   dev_strategy, update_table, ret_table,
                                   'cache/{}/dev_pred_{}.pkl'.format(args.name, i))])
                dev_loss, dev_accu, dev_step_info = info[1]
                # draw_info(dev_loss, dev_accu, dev_step_info, 'dev', i)
                # draw_dev_info(dev_loss, dev_accu, None, 'dev', self.writer, i)
                draw_info(dev_loss, dev_accu, dev_step_info, 'dev', i)

            # print('[DEBUG] {} time {}s.'.format('valid', time.time() - cur_t))
            # cur_t = time.time()

            if i == 0:
                print('train_step_info:', train_step_info[2])
                # print('dev_step_info:', dev_step_info[2])
            print('[train{}/{}]\t time:{:.2f}s.'.format(i+1, args.epochs, time.time()-cur_t))
            cur_t = time.time()
            if update_table['id']:
                print('\t<identity> train loss{:.5f} accu{:.5f}, valid loss{:.5f} accu{:.5f}, '
                      .format(train_loss[0], train_accu[0], dev_loss[0], dev_accu[0]))
            if update_table['tom']:
                print('\t<tom> train ploss{:.5f} accu{:.5f}, valid ploss{:.5f} accu{:.5f}, '.
                      format(train_loss[2], train_accu[1], dev_loss[2], dev_accu[1]))


            if not update_table['tom']:
                # When only update id
                score = dev_accu[0]
                score_type = 'accu'
            else:
                score = dev_loss[2]
                score_type = 'loss'

            if (i+1)%30 == 0:
                # Only update best model
                worker.local_send(['save_best_model', (i, score, score_type, True)])

                # print('[DEBUG] {} time {}s.'.format('dump_model', time.time() - cur_t))
                # cur_t = time.time()

            elif (i+1)%100 == 0:
                worker.local_send(['save_best_model', (i, score, score_type, False)])

                # print('[DEBUG] {} time {}s.'.format('dump_model', time.time() - cur_t))
                # cur_t = time.time()

    def _log_policy(self, examples, dump_result):
        policies = [{
            'i_policy': [], 'p_policy': []
        }, {
            'i_policy': [], 'p_policy': []
        }]
        for ex in examples:
            for e in ex.events:
                i = e.agent
                odata = e.metadata['output_data']
                policies[i]['i_policy'].append(odata['policy'].reshape(1, -1))
                if odata.get('p_policy') is not None:
                    policies[i]['p_policy'].append(odata['p_policy'].reshape(1, -1))

        for i in range(2):
            for k in policies[i]:
                if len(policies[i][k]) > 0:
                    policies[i][k] = torch.cat(policies[i][k], dim=0).mean(dim=0, keepdim=True)
                    if self.policies_log[i].get(k) is None:
                        self.policies_log[i][k] = []
                    self.policies_log[i][k].append(policies[i][k])
                    if dump_result:
                        logger = logging.getLogger('agent{}_plog_{}'.format(i, k))
                        tmp = torch.cat(self.policies_log[i][k], dim=0).mean(dim=0)
                        # tensor([x, x, x])
                        logger.info(str(tmp.data)[8:-2].replace("        ", "").replace("\n", ""))

    def _init_policy_logfiles(self, logdir):
        formatter = logging.Formatter('%(message)s')
        # stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(logging.DEBUG)
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)
        for i in range(2):
            for name in ['i_policy', 'p_policy']:
                file_handler = logging.FileHandler(os.path.join(logdir, 'agent{}_plog_{}.log'.format(i, name)))
                file_handler.setLevel(level=logging.INFO)
                file_handler.setFormatter(formatter)

                logger = logging.getLogger('agent{}_plog_{}'.format(i, name))
                logger.setLevel(level=logging.INFO)
                logger.addHandler(file_handler)


    def learn_dsac(self):
        args = self.args
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0
        self.update_worker_list()
        worker = self.worker_conn[0]
        max_epoch = args.epochs
        save_every = max(50, max_epoch // 100)
        if args.debug:
            save_every = 1

        device = 'cpu'
        if len(args.gpuid) > 0:
            device = "cuda:{}".format(args.gpuid[0])
        self._init_policy_logfiles('logs/' + args.name)
        sample_size = 32
        train_size = 128
        from onmt import replay_memory
        my_replay_buffer = replay_memory.ReplayBuffer(max_size=args.replay_buffer)
        is_train_critic=0
        is_pretrain = True
        if self.args.self_play:
            self.trainer.trainer.opponents_pool['self']['self_agent'] = self.trainer.systems[2]
        for epoch in range(max_epoch):
            last_time = time.time()
            if args.self_play:
                info = worker.send(['simulate_pfsp', epoch, sample_size, sample_size])
            else:
                info = worker.send(['simulate', epoch, sample_size, sample_size])
            _batch_iters, batch_info, example, v_str = pkl.loads(info[1])
            _rewards, strategies = batch_info

            self.add_to_replay_buffer(_batch_iters, _rewards, my_replay_buffer)
            # self._log_policy(example, (epoch + 1) % save_every == 0)
            if self.args.self_play:
                hard_update(self.trainer.trainer.opponents_pool['self']['self_agent'].env.dsac.actor_model,
                            self.trainer.trainer.policy)
            tt = time.time()
            loss = None
            if len(my_replay_buffer.storage) > train_size:

                batch_experience = my_replay_buffer.sample(train_size)
                if  is_pretrain:
                    update_critic=True
                    update_policy=False
                    is_train_critic += 1
                    if is_train_critic == 50:
                        my_replay_buffer.clear()
                        is_pretrain = False
                else:
                    update_critic = True
                    update_policy = True

                info = worker.local_send(
                    ['train_dsac', (batch_experience, update_policy, update_critic)])
                zf1_loss, zf2_loss, policy_loss, alpha_intent, alpha_price, q_loss, kl_loss = info[1]
                self.writer.add_scalar("zf1_loss", zf1_loss.item(), epoch)
                self.writer.add_scalar("zf2_loss", zf2_loss.item(), epoch)
                if policy_loss is not None:
                    self.writer.add_scalar("policy_loss", policy_loss.item(), epoch)
                    self.writer.add_scalar("alpha_intent", alpha_intent.item(), epoch)
                    self.writer.add_scalar("alpha_price", alpha_price.item(), epoch)
                    self.writer.add_scalar("q_loss", q_loss.item(), epoch)
                    self.writer.add_scalar("kl_loss", kl_loss.item(), epoch)
                print('train time:', time.time() - tt)

            # Draw outputs on the tensorboard


            print('\t<train> reward{:.3f}, {:.3f} '
                  .format(np.mean(_rewards[0]), np.mean(_rewards[1]),))
            print("replau buffer size:", len(my_replay_buffer.storage))

            def get_result_dict(d):
                ret = {}
                if d.get('reward') is None:
                    num = 0
                else:
                    num = len(d.get('reward'))
                for k in d:
                    ret[k] = np.mean(d[k])
                    ret[k + '_std'] = np.std(d[k])

                return ret, num

            if (epoch + 1) % 10 == 0 and not self.args.self_play:
                print("evaluate at epoch:", epoch)
                infos = worker.local_send(['valid', (0, 50)])
                examples = infos[1][1]
                eval_dict, _ = worker.local_send(['get_eval_dict', (examples, [])])[1]
                result = get_result_dict(eval_dict)[0]
                self.writer.add_scalar('Evaluate agent dsac of  reward mean', result.get('reward'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  reward std', result.get('reward_std'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  success rate mean', result.get('success_rate'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  success rate std', result.get('success_rate_std'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  utility mean', result.get('utility'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  utility std', result.get('utility_std'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  length mean', result.get('length'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  length std', result.get('length_std'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  fairness mean', result.get('fairness'), epoch)
                self.writer.add_scalar('Evaluate agent dsac of  fairness std', result.get('fairness_std'), epoch)

            if epoch >= 100 and (epoch + 1) % 10 == 0 and self.args.self_play:
                print("evaluate using self_play at epoch:", epoch)
                prob = [self.trainer.trainer.score_table["sl"][item]['win_rate'][-1] for item in
                        self.trainer.trainer.score_table["sl"].keys()]
                prob = self.trainer.trainer.calculate_pfsp_prob(prob)
                print(prob)
                #["low", "sigmoid", "persuaded", "convex", "high", "wait", "decay", "concave"]
                # for kind in ["self"]:
                for kind in self.trainer.trainer.opponents_pool.keys():
                    for opponent in self.trainer.trainer.opponents_pool[kind].keys():
                        print("evaluate with ",opponent)
                        infos = worker.local_send(['valid_sp', (0, 20, self.trainer.trainer.opponents_pool[kind][opponent])])
                        examples = infos[1][0]
                        eval_dict = worker.local_send(['get_eval_dict_sp', (examples, opponent)])[1]
                        result = get_result_dict(eval_dict)[0]
                        self.writer.add_scalar('Evaluate agent dsac against {} of  reward mean'.format(opponent), result.get('reward'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  reward std'.format(opponent), result.get('reward_std'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  success rate mean'.format(opponent), result.get('success_rate'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  success rate std'.format(opponent), result.get('success_rate_std'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  utility mean'.format(opponent), result.get('utility'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  utility std'.format(opponent), result.get('utility_std'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  length mean'.format(opponent), result.get('length'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  length std'.format(opponent), result.get('length_std'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  fairness mean'.format(opponent), result.get('fairness'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  fairness std'.format(opponent), result.get('fairness_std'), epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  score'.format(opponent), self.trainer.trainer.score_table[kind][opponent]['score'][-1], epoch)
                        self.writer.add_scalar('Evaluate agent dsac against {} of  win_rate'.format(opponent), self.trainer.trainer.score_table[kind][opponent]['win_rate'][-1], epoch)
                self.writer.add_scalar('Evaluate agent dsac avg_score',
                                       self.trainer.trainer.get_avg_score(), epoch)

                if self.check_total_performance():
                    from core.price_tracker import PriceTracker
                    lexicon = PriceTracker(args.price_tracker_model)
                    schema = Schema(args.schema_path)
                    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
                    valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)
                    from systems.dsac_neural_system import DSACNeuralSystem
                    self.save_dsac_model(epoch, args)
                    dsac_load_dict = self.get_dsac_load_dict(epoch, args)
                    self.trainer.trainer.opponents_pool['rl']["rl_"+str(len(self.trainer.trainer.opponents_pool['rl']))] = \
                        DSACNeuralSystem(args, schema, lexicon, args.agent_checkpoints[0], False, name='pt-neural-dsac', id=id, dsac_load_dict=dsac_load_dict)
                    self.trainer.trainer.score_table['rl']["rl_"+str(len( self.trainer.trainer.score_table['rl']))] = {'success_rate': [0.5], 'utility': [0.5], 'length': [10], 'fairness': [1], 'score': [1.46], 'win_rate':[0.5]}



            if (epoch + 1) % save_every == 0:
                print("saving dsac model")
                self.save_dsac_model(epoch, args)
                self.dump_examples(example, v_str, epoch, 'train')
                
            print('=' * 5 + ' [Epoch {}/{} for {:.3f}s.]'.format(epoch + 1, max_epoch, time.time() - last_time))

    def check_total_performance(self):
        flag1 = self.check_each_performance('sl')
        flag2 = self.check_each_performance('rl')
        return flag1 and flag2

    def check_each_performance(self, opponent_kind):
        if len(self.trainer.trainer.score_table[opponent_kind].keys()) == 0:
            return True
        flag1 = False
        flag2 = False
        flag3 = False
        win_rate = []
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        for opponent in self.trainer.trainer.score_table[opponent_kind].keys():
            win_rate.append(self.trainer.trainer.score_table[opponent_kind][opponent]['win_rate'][-1])
            if self.trainer.trainer.score_table[opponent_kind][opponent]['win_rate'][-1] > 0.6:
                cnt1 += 1
            if self.trainer.trainer.score_table[opponent_kind][opponent]['win_rate'][-1] > 0.7:
                cnt2 += 1
            if self.trainer.trainer.score_table[opponent_kind][opponent]['win_rate'][-1] > 0.8:
                cnt3 += 1
        print("win_rate", win_rate)
        print("self.trainer.trainer.score_table[opponent_kind].keys()",self.trainer.trainer.score_table[opponent_kind].keys())
        l = len(self.trainer.trainer.score_table[opponent_kind].keys())
        if cnt1 > 0 and cnt1 >= l - 1:
            flag1 = True
        if cnt2 > 0 and cnt2 >= l - 2:
            flag2 = True
        if cnt3 > 0 and cnt3 >= l - 3:
            flag3 = True
        return flag1 or flag2 or flag3



    def save_dsac_model(self, epoch, args):
        torch.save(self.trainer.trainer.policy.state_dict(), args.model_path + "/actor_seed" + str(args.seed) + "_"+str(epoch + 1)+".pth")
        torch.save(self.trainer.trainer.zf1.state_dict(),args.model_path + "/zf1_seed" + str(args.seed) + "_"+str(epoch + 1)+".pth")
        torch.save(self.trainer.trainer.zf1.state_dict(),args.model_path + "/zf2_seed" + str(args.seed) + "_" + str(epoch + 1) + ".pth")

    def get_dsac_load_dict(self, epoch, args):
        dsac_load_dict={'actor':args.model_path + "/actor_seed" + str(args.seed) + "_"+str(epoch + 1)+".pth",
                        'zf1':args.model_path + "/zf1_seed" + str(args.seed) + "_"+str(epoch + 1)+".pth",
                       'zf2':args.model_path + "/zf2_seed" + str(args.seed) + "_" + str(epoch + 1) + ".pth"}
        return dsac_load_dict

    def learn(self):
        args = self.args
        rewards = [None] * 2
        s_rewards = [None] * 2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        save_every = 100

        history_train_losses = [[], []]

        batch_size = 50

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)
        num_worker = self.update_worker_list()

        worker = self.worker_conn[0]
        max_epoch = args.num_dialogues // batch_size
        max_epoch = args.epochs
        batch_size = args.batch_size

        save_every = max(50, max_epoch // 100)
        report_every = max(1, max_epoch // 100)

        if args.debug:
            save_every = 1

        device = 'cpu'
        if len(args.gpuid) > 0:
            device = "cuda:{}".format(args.gpuid[0])

        policy_buffer = ReplayBuffer.get_instance('policy')
        value_buffer = ReplayBuffer.get_instance('value')
        self._init_policy_logfiles('logs/' + args.name)

        sample_size = 128
        train_size = 128

        for epoch in range(max_epoch):
            last_time = time.time()
            policy_buffer.empty()
            tt = time.time()
            info = worker.send(['simulate', epoch, sample_size, sample_size])
            _batch_iters, batch_info, example, v_str = pkl.loads(info[1])
            _rewards, strategies = batch_info
            self._log_policy(example, (epoch+1) % save_every == 0)

            policy_buffer.add_batch_iters(_batch_iters[0],
                                          add_dict={'reward': _rewards[0], 'strategy': strategies[0]})
            value_buffer.add_batch_iters(_batch_iters[0],
                                         add_dict={'reward': _rewards[0], 'strategy': strategies[0]})

            tt = time.time()
            value_update = min(value_buffer.size//train_size, 5)
            for i in range(value_update):
                batch_iters, _, ret_add = value_buffer.sample_batch(train_size, add_info={'reward'}, to_device=device)
                worker.local_send(
                    ['train', (epoch, batch_iters, ret_add['reward'], 'fix_policy')])

            batch_iters, _, ret_add = policy_buffer.sample_batch(train_size, add_info={'reward'}, to_device=device)

            info = worker.local_send(
                ['train', (epoch, batch_iters, ret_add['reward'], '')])
            loss = info[1]
            print('train time:', time.time()-tt)

            # Draw outputs on the tensorboard
            self._draw_tensorboard((epoch + 1) , [[loss], []],
                                   _rewards)

            print('\t<train> reward{:.3f}, {:.3f} pg_loss {:.5f}, value_loss {:.5f}, value_update {}'
                  .format(np.mean(_rewards[0]), np.mean(_rewards[1]), loss['pg_loss'][0,0], loss['value_loss'][0,0], value_update))

            if (epoch+1)%save_every == 0:
                self._dump_buffer(value_buffer, epoch+1)

                self.dump_examples(example, v_str, epoch, 'train')
                valid_info = worker.local_send(['valid', (0, 200)])
                valid_stats, example, v_str = valid_info[1]
                self.dump_examples(example, v_str, epoch, 'dev')

                valid_reward = [vs.mean_reward() for vs in valid_stats]
                self._draw_tensorboard_valid((epoch + 1), valid_reward)
                print('\t<valid> reward{:.3f}, {:.3f}'.format(valid_reward[0], valid_reward[1]))
                worker.local_send(['save_model', (epoch, valid_reward[0], 'reward')])
            print('=' * 5 + ' [Epoch {}/{} for {:.3f}s.]'.format(epoch+1, max_epoch, time.time() - last_time))
            # # Save model
            # if (i + 1) % save_every == 0:
            #     # TODO: valid in dev set
            #     valid_stats, _, _ = self.validate(args, 50 if args.only_run else 200)
            #     if not args.only_run:
            #         self.drop_checkpoint(args, i + 1, valid_stats,
            #                              model_opt=self.agents[self.training_agent].env.model_args)
            #         if args.update_oppo:
            #             self.update_opponent(['policy', 'critic'])
            #     else:
            #         print('valid ', valid_stats.str_loss())

    def add_to_replay_buffer(self, batch_iters, rewards, replay_buffer):
        for idx, batch_iter in enumerate(batch_iters[0]):
            for i in range(len(batch_iter)):
                transition = []
                transition.extend([batch_iter[i].state_0.squeeze(), batch_iter[i].state_1.squeeze(), batch_iter[i].uttr[0].squeeze()])
                transition.extend([batch_iter[i].act_0.squeeze(), batch_iter[i].act_2.squeeze(), batch_iter[i].act_3.squeeze()])
                if i != len(batch_iter) - 1:
                    transition.extend([batch_iter[i + 1].state_0.squeeze(), batch_iter[i + 1].state_1.squeeze(), batch_iter[i + 1].uttr[0].squeeze()])
                    if batch_iter[i+1].act_0[0][0].item() == batch_iter[i+1].acc_idx or batch_iter[i+1].act_0[0][0].item() == batch_iter[i+1].rej_idx:
                        transition.append(rewards[0][idx]) # reward
                        transition.append(1.0) # done bool
                    else:
                        transition.append(0.0) # reward
                        transition.append(0.0) # done bool
                else:
                    if batch_iter[i].act_0[0][0].item() == batch_iter[i].offer_idx:
                        transition.extend([batch_iter[i].state_0.squeeze(), batch_iter[i].state_1.squeeze(), batch_iter[i].uttr[0].squeeze()])
                        transition.append(rewards[0][idx]) # reward
                        transition.append(1.0) # done bool
                    else:
                        continue

                replay_buffer.add(tuple(transition))


    def _dump_buffer(self, buffer, epoch, ):
        args = self.args
        path_pkl = '{root}/{model}_buffer{epoch}.pkl'.format(
            root=args.model_path,
            model=args.name,
            epoch=epoch)
        print('Save buffer at {}.'.format(path_pkl))
        with open(path_pkl, 'wb') as f:
            pkl.dump(buffer, f)
        # with open(path_pkl, 'rb') as f:
        #     bf = pkl.load(f)

    def run(self):
        # deprecated
        # self.run_local_workers()
        args = self.args
        rewards = [None] * 2
        s_rewards = [None] * 2
        tensorboard_every = 1
        save_every = 50

        history_train_losses = [[], []]

        batch_size = 100

        pretrain_rounds = 3

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        max_epoch = args.num_dialogues // batch_size
        epoch = 0
        data_size = 0

        all_rewards = [[], []]

        num_worker = self.update_worker_list()
        last_time = time.time()
        for epoch in range(args.start_epoch, max_epoch):
            batches = []
            rewards = [[], []]

            task_lists = self.allocate_tasks(num_worker, batch_size)

            # Use workers to get trajectories
            train_examples = []
            train_ex_str = []
            for i, w in enumerate(self.worker_conn):
                info = w.send(['simulate', epoch, batch_size, task_lists[i]])
                if info[0] != 'done':
                    print('Error on {}: {}.'.format(i, info))
                data = pkl.loads(info[1])
                batches += data[0]
                rewards[0] += data[1][0]
                rewards[1] += data[1][1]
                train_examples += data[2]
                train_ex_str += data[3]

            self.dump_examples(train_examples, train_ex_str, epoch)

            # For debug
            print("rewards:", np.mean(rewards[0]), np.mean(rewards[1]))
            print("rewards_num:", len(rewards[0]), len(rewards[1]))

            # Train the model
            train_info = self.worker_conn[0].send(['train', pkl.dumps((epoch, batches, rewards[0], self.args.train_mode))])
            if train_info[0] != 'done':
                print('Error on {}: {}.'.format(i, train_info))

            # Draw outputs on the tensorboard
            self._draw_tensorboard((epoch + 1) * batch_size, [[pkl.loads(train_info[1])], []],
                                   rewards)

            # Get new model from trainer

            info = self.worker_conn[0].send(['fetch_model', 0])
            data = info[1]

            # Save local checkpoint

            # Update all the worker
            for i, w in enumerate(self.worker_conn):
                if i == 0:
                    continue
                w.send(['update_model', 0, data])

            # for i, w in enumerate(self.worker_conn):
            #     if i == 0:
            #         continue
            #     w.recv()

            # Valid new model
            task_lists = self.allocate_tasks(num_worker, 50)
            now = 0

            valid_stats = [RLStatistics(), RLStatistics()]
            valid_examples = []
            valid_ex_str = []
            for i, w in enumerate(self.worker_conn):
                valid_info = w.send(['valid', (now, task_lists[i])])
                now += task_lists[i]
                valid_info[1] = pkl.loads(valid_info[1])
                for j in range(2):
                    valid_stats[j].update(valid_info[1][0][j])
                valid_examples += valid_info[1][1]
                valid_ex_str += valid_info[1][2]

            self.dump_examples(valid_examples, valid_ex_str, epoch, 'dev')
            # Save the model
            self.worker_conn[0].send(['save_model', pkl.dumps((epoch, valid_stats[0]))])
            # self.worker_conn[0].recv()

            # Draw dev rewards on tensorboard
            dev_rewards = [valid_stats[j].mean_reward() for j in range(2)]
            self._draw_tensorboard_valid((epoch + 1) * batch_size, dev_rewards)

            print('=' * 5 + ' [Epoch {} for {:.3f}s.]'.format(epoch, time.time() - last_time))
            last_time = time.time()

        self.quit_all_workers()
        self.join_local_workers()

    def quit_all_workers(self):
        for w in self.worker_conn:
            w.send(['quit'])

    def join_local_workers(self):
        # for w in self.local_workers:
        #     w.join()
        pass