# Data: _batch_iters, (_rewards, strategies), examples, verbose_strs
# Out: (pred_identity, pred_intent, pred_price, strategies)

import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.controller import Controller
from .utterance import UtteranceBuilder

from tensorboardX import SummaryWriter
import pickle as pkl

from neural.batcher_rl import RLBatch, RawBatch, ToMBatch

from neural.rl_trainer import RLTrainer as BaseTrainer
from neural.sl_trainer import Statistics, SimpleLoss
from neural.generator import LFSampler

import math, time, sys


class RLStatistics(Statistics):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, reward=0, n_words=0):
        self.loss = loss
        self.n_words = n_words
        self.n_src_words = 0
        self.reward=reward
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.reward += stat.reward

    def mean_loss(self):
        return self.loss / self.n_words

    def mean_reward(self):
        return self.reward / self.n_words

    def elapsed_time(self):
        return time.time() - self.start_time

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def str_loss(self):
        return "loss: %6.4f reward: %6.4f;" % (self.mean_loss(), self.mean_reward())

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d;" + self.str_loss() +
               "%6.0f s elapsed") %
              (epoch, batch,  n_batches,
               time.time() - start))
        sys.stdout.flush()

class SimpleCriticLoss(nn.Module):
    def __init__(self):
        super(SimpleCriticLoss, self).__init__()
        self.criterion = nn.MSELoss()

    # def _get_correct_num(self, enc_policy, tgt_intents):
    #     enc_policy = enc_policy.argmax(dim=1)
    #     tmp = (enc_policy == tgt_intents).cpu().numpy()
    #     tgt = tgt_intents.data.cpu().numpy()
    #     tmp[tgt==19] = 1
    #     import numpy as np
    #     return np.sum(tmp)

    def forward(self, pred, oracle, pmask=None):
        loss = self.criterion(pred, oracle)
        stats = self._stats(loss, pred.shape[0])
        return loss, stats

    def _stats(self, loss, data_num):
        return RLStatistics(loss=loss.item(), n_words=data_num)

class RLTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin',
                 cuda=False, args=None):
        super(RLTrainer, self).__init__(agents, scenarios, train_loss, optim,
                                        training_agent, reward_func, cuda, args)
        # print('training_agent', training_agent)

        self.critic = agents[training_agent].env.critic
        self.tom = agents[training_agent].env.tom_model
        self.vocab = agents[training_agent].env.vocab
        self.lf_vocab = agents[training_agent].env.lf_vocab
        self.model_type = args.model_type
        self.use_utterance = False
        self.tom_identity_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.hidden_vec = None

    def _run_batch_a2c(self, batch):
        value = self._run_batch_critic(batch)
        policy, price = self._run_batch(batch)
        # print('max price', torch.max(price))
        return value, policy, price

    def _run_batch_tom_identity(self, batch, hidden_state, only_identity=False, id_gt=False):
        if id_gt:
            id_gt = batch.strategy
        else:
            id_gt = None
        if only_identity:
            identity, next_hidden = \
                self.tom.encoder.identity(batch.identity_state, batch.extra, hidden_state, uttr=batch.uttr)
            predictions = None
        else:
            output = self.tom(batch.uttr, batch.identity_state, batch.state,
                              batch.extra, hidden_state, id_gt)
            if len(output) == 3:
                predictions, next_hidden, identity = output
            else:
                predictions, next_hidden = output
                identity = None
        return predictions, next_hidden, identity

    def _tom_gradient_accumulation(self, batch_iter, strategy, model, ret_table, id_gt=False):
        model.train()

        h = None
        identity_loss = []
        tom_intent_loss = []
        tom_price_loss = []
        identity_accu = []
        identity_accu2 = []
        tom_intent_accu = []
        strategies = []

        pred_intent = []
        pred_price = []
        pred_identity = []

        for i, batch in enumerate(batch_iter):
            tom_batch = ToMBatch.from_raw(batch, strategy[:batch.size])
            if h is not None:
                if isinstance(h, tuple):
                    h = tuple(map(lambda x: x[:batch.size, :], h))
                    # h = (h[0][:batch.size, :], h[1][:batch.size, :])
                elif isinstance(h, torch.Tensor):
                    h = h[:batch.size, :]
            pred, h, identity = self._run_batch_tom_identity(tom_batch, hidden_state=h,
                                                             only_identity=(not ret_table['tom']), id_gt=id_gt)

            self.hidden_vec.append(self.tom.hidden_vec)
            s = np.array([strategy[:batch.size]]).T
            s = np.concatenate([s, tom_batch.identity_state.cpu().data.numpy(), tom_batch.extra.cpu().data.numpy(),], axis=1)
            self.hidden_stra.append(s)

            # Identity Loss
            if ret_table['id']:
                s = torch.tensor(strategy[:batch.size], dtype=torch.int64, device=identity.device)
                loss = self.tom_identity_loss(identity, s)
                # accu = torch.gather(torch.softmax(identity, dim=1), 1, s.reshape(-1, 1))
                id_p = torch.softmax(identity, dim=1)
                accu = id_p.argmax(dim=-1).reshape(-1, 1) == s.reshape(-1, 1)
                accu = accu.to(dtype=torch.float32)
                accu2 = (id_p.topk(3, dim=-1).indices == s.reshape(-1, 1)).max(dim=-1).values.reshape(-1, 1)
                accu2 = accu2.to(dtype=torch.float32)
                identity_loss.append(loss.reshape(-1))
                identity_accu.append(accu.reshape(-1))
                identity_accu2.append(accu2.reshape(-1))
                pred_identity.append(identity.reshape(1, -1).detach())

            # ToM Loss
            if ret_table['tom']:
                intent, price = pred
                loss0, loss1, batch_stats = self._compute_loss(tom_batch, policy=intent, price=price, loss=self.tom_loss)
                intent_accu = torch.gather(torch.softmax(intent, dim=1), 1, tom_batch.act_intent.reshape(-1, 1))
                tom_intent_accu.append(intent_accu)
                tom_intent_loss.append(loss0.reshape(-1))
                tom_price_loss.append(loss1.reshape(-1))
                pred_intent.append(intent.reshape(1, -1).detach())
                pred_price.append(price.reshape(1, -1).detach())

            # losses.append(loss.reshape(-1))
            # accus.append(accu.reshape(-1))
            # strategies.append(s.detach())

            # preds.append(pred.reshape(1, -1).detach())


        # preds = torch.cat(preds, dim=0)
        # strategy = torch.tensor([strategy]*preds.shape[0], dtype=torch.int64, device=preds.device)
        # (-1,), (-1, 1) -> (-1,) *2
        # print('loss & accu:', loss, accu)
        return {'id':[identity_loss], 'tom':[tom_intent_loss, tom_price_loss]}, {'id':[identity_accu, identity_accu2], 'tom':[tom_intent_accu], }, \
               (pred_identity, pred_intent, pred_price, strategies)

    def _sort_merge_batch(self, batch_iters, batch_size, device=None):
        sorted_id = [i for i in range(len(batch_iters))]
        sorted_id.sort(key=lambda i: len(batch_iters[i]), reverse=True)
        batch_iters = sorted(batch_iters, reverse=True, key=lambda l: len(l))
        batch_length = [len(b) for b in batch_iters]

        if device is None:
            if self.cuda:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        def merge_batch(one_batch):
            batches = [[] for i in range(len(one_batch[0]))]
            for bi in one_batch:
                for i, b in enumerate(bi):
                    batches[i].append(b)
            for i, b in enumerate(batches):
                # print('merge batch:', i, len(b))
                batches[i] = RawBatch.merge(b)
                batches[i].to(device)
            return batches

        # Split by size
        right = 0
        bs, ids, bl = [], [], []
        while True:
            left = right
            right = min(right+batch_size, len(batch_iters))
            # print('merge: ', left, right)
            bs.append(merge_batch(batch_iters[left: right]))
            ids.append(sorted_id[left: right])
            bl.append(batch_length[left: right])
            if right >= len(batch_iters):
                break

        return bs, ids, bl

    @staticmethod
    def split_by_strategy(t, s, s_num=5):
        if s is None:
            return t
        s = torch.tensor(s, dtype=torch.uint8, device=t.device).reshape(-1, 1)
        ret = [0]*s_num
        for i in range(s_num):
            ret[i] = torch.masked_select(t, s == i)
        # ret[0]=torch.masked_select(t, s)
        # ret[1]=torch.masked_select(t, 1-s)
        return ret

    def add_strategy_in_language(self, batch_iters, strategies):
        for i in range(2):
            if i == 0:
                continue
            # for each dialogue
            for j in range(len(batch_iters[i])):
                c = self.vocab.size-1-strategies[i][j]
                # for each sentences
                for k, b in enumerate(batch_iters[i][j]):
                    if random.randint(0, 5) > 0:
                        continue
                    tmp = b.uttr[0].cpu().numpy()
                    l = np.prod(tmp.shape)
                    tmp = np.insert(tmp, random.randint(2, l-1), c, axis=1)
                    b.uttr[0] = torch.tensor(tmp, device=b.uttr[0].device)

    def update_tom(self, args, batch_iters, strategy, model,
                   update_table=None, ret_table=None, dump_name=None):
        # print('optim', type(self.optim['tom']))
        cur_t = time.time()
        if not update_table:
            update_table = {'id': True, 'tom': True}
        if not ret_table:
            ret_table = {'id': True, 'tom': True}

        batch_iters, sorted_id, batch_length = batch_iters
        # print('merge batch: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()

        # TODO: split_by_strategy?
        split_by_strategy = False

        model.zero_grad()
        loss = {'id': [[]], 'tom': [[], []]}
        accu = {'id': [[], []], 'tom': [[]], }
        step_loss = {'id': [[]], 'tom': [[], []]}
        step_accu = {'id': [[], []], 'tom': [[]], }
        # step_loss = [[] for i in range(20)]
        # step_accu = [[] for i in range(20)]
        output_data = []

        def add_list(step, one, s=None):
            for j, o in enumerate(one):
                if isinstance(o, list):
                    for k in range(len(o)):
                        if k >= len(step[j]):
                            step[j].append([])
                        step[j][k].append(self.split_by_strategy(o[k], s))
                else:
                    step[j].append(o)

        self.hidden_vec = []
        self.hidden_stra = []

        for i, b in enumerate(batch_iters):
            stra = [strategy[j] for j in sorted_id[i]]
            l, a, logs = self._tom_gradient_accumulation(b, stra, model, ret_table=ret_table, id_gt=args.idgt)

            # print('[DEBUG] {} time {}s.'.format('grad_accu', time.time() - cur_t))
            # cur_t = time.time()

            output_data.append(logs)
            # weight = torch.ones_like(l, device=l.device)
            # for j in range(5):
            #     if j < l.shape[0]:
            #         weight[j] = 5-j
            # l = l.mul(weight)
            # for j, ll in enumerate(l):
            #     if j == len(l)-1:
            #         loss.append(ll)
            #     else:
            #         if l[j+1].shape[0] > ll.shape[0]:
            #             loss.append(ll[l[j+1].shape[0]:])
            if not split_by_strategy:
                stra = None
            # loss
            # l['tom']=[loss_intent, loss_price]
            # l['tom'][0] = [Tensor, Tensor, ...], whose length is number of steps
            # loss['tom'] = [loss_intent, loss_price]
            # loss['tom'] = [Tensor, ...], whose length is number of batch
            for key in l:
                l_key = l[key]
                if not ret_table[key]: continue
                for j, ll in enumerate(l_key):
                    tmp = torch.cat(ll, dim=0)
                    loss[key][j].append(self.split_by_strategy(tmp, stra))
                add_list(step_loss[key], l_key, stra)

            for key in a:
                a_key = a[key]
                if not ret_table[key]: continue
                for j, aa in enumerate(a_key):
                    tmp = torch.cat(aa, dim=0)
                    accu[key][j].append(self.split_by_strategy(tmp, stra))
                add_list(step_accu[key], a_key, stra)

            # print('[DEBUG] {} time {}s.'.format('append', time.time() - cur_t))
            # cur_t = time.time()

        # print('calculate loss: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()
        step_num = None
        for key in ['id', 'tom']:
            if not ret_table[key]:
                continue

            if step_num is None:
                if split_by_strategy:
                    if ret_table['id']:
                        step_num = [[np.sum([dd[i].shape[0] for dd in d]) for d in step_loss[key][0]]
                                    for i in range(2)]
                    else:

                        step_num = [[np.sum([dd[i].shape[0] for dd in d]) for d in step_loss[key][0]]
                                    for i in range(2)]
                else:
                    step_num = [np.sum([dd.shape[0] for dd in d]) for d in step_loss[key][0]]

            for i, l in enumerate(loss[key]):
                loss[key][i] = torch.cat(l, dim=0).mean()
                if split_by_strategy:
                    step_loss[key][i] = [[torch.cat([dd[j] for dd in d], dim=0).mean().item() if len(d) > 0 else None for d in step_loss[key][i]]
                                    for j in range(2)]
                else:
                    step_loss[key][i] = [torch.cat(d, dim=0).mean().item() if len(d)>0 else None for d in step_loss[key][i]]

            for i, a in enumerate(accu[key]):
                accu[key][i] = torch.cat(a, dim=0).mean().item()
                step_accu[key][i] = [torch.cat(d, dim=0).mean().item() if len(d)>0 else None for d in step_accu[key][i]]

        # print('returen infos: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()

        # dump output data
        if dump_name is not None:
            with open(dump_name, 'wb') as f:
                pkl.dump(output_data, f)

        # print('[DEBUG] {} time {}s.'.format('mean & dump', time.time() - cur_t))
        # cur_t = time.time()

        # update
        if update_table['id']:
            loss['id'][0].backward()
            if self.optim.get('tom_identity') is not None:
                self.optim['tom_identity'].step()
            else:
                print('[Warning] update identity, but no identity exists.')

        if ret_table['tom']:
            loss['tom'][1] *= 1000

        if update_table['tom']:
            l = loss['tom'][0] + loss['tom'][1]
            l.backward()
            self.optim['tom'].step()

        # print('[DEBUG] {} time {}s.'.format('backward', time.time() - cur_t))
        # cur_t = time.time()

        for key in ['id', 'tom']:
            for i, l in enumerate(loss[key]):
                if isinstance(loss[key][i], torch.Tensor):
                    loss[key][i] = loss[key][i].item()
                else:
                    loss[key][i] = None

        # print('udpate model: {}s.'.format(time.time() - cur_t))
        # cur_t = time.time()
        loss = loss['id'] + loss['tom']
        accu = accu['id'][:1] + accu['tom'] + accu['id'][1:]
        step_loss = step_loss['id'] + step_loss['tom']
        step_accu = step_accu['id'][:1] + step_accu['tom'] + step_accu['id'][1:]

        return loss, \
               accu, (step_loss, step_accu, step_num)

    def _gradient_accumulation(self, batch_iter, reward, model, critic, discount=1):
        # Compute losses
        model.train()
        critic.train()

        values = []
        losses = [[], []]
        ents = [[], []]

        # batch_iter gives a dialogue
        policy_stats = Statistics()
        # For value: deprecated
        for_value = False

        # In one batch, from sentence 1 to n.
        for i, batch in enumerate(batch_iter):
            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # batch.mask_last_price()
            rlbatch = RLBatch.from_raw(batch, None, None)
            value, i_pred, p_pred = self._run_batch_a2c(rlbatch)

            # The last sentence may only need to predict value?
            # if not for_value:
                # intent_loss, price_stats, batch_stats = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.train_loss)
            # print('it', i_pred, rlbatch.act_intent)
            # print('price', p_pred, rlbatch.act_price)
            intent_loss = F.cross_entropy(i_pred, rlbatch.act_intent.reshape(-1), reduction='none')
            pact_loss = F.cross_entropy(p_pred, rlbatch.act_price.reshape(-1), reduction='none')
                # print('policy_loss is:', policy_loss)
                # policy_stats.update(pl_stats)

            # entropy_loss, _ = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.entropy_loss)

            intent_ent = self.entropy_loss(i_pred)
            if torch.isnan(intent_ent.mean()):
                isnan = torch.isnan(intent_ent.mean()).reshape(-1)
                intent_ent = intent_ent.reshape(-1)
                for j in range(isnan.shape[0]):
                    if isnan[j] == 1:
                        print('nan: ', i_pred[j])
                        print('nan2: ', rlbatch.act_intent.reshape(-1)[j], intent_loss.reshape(-1)[j])
                quit()
            pact_ent = self.entropy_loss(p_pred)
            pact_loss = pact_loss.reshape(-1, 1)*rlbatch.act_price_mask
            pact_ent = pact_ent.reshape(-1, 1)*rlbatch.act_price_mask

            # policy_loss = intent_loss + pact_loss
            # entropy_loss = intent_ent + pact_ent


            # penalty = ((price-1)**2).mul((price>2).float()) + ((price-0)**2).mul((price<0.5).float())
            # penalty = ((price > 2).float()).mul((price - 1) ** 2) + ((price < 0.5).float()).mul((price - 0) ** 2)
            # penalty = ((price > 2).float()).mul(0.1) + ((price < 0.5).float()).mul(0.1)
            # penalty = torch.zeros_like(price, device=price.device)

            # if not for_value:
            # penalties.append(penalty.view(-1))
            losses[0].append(intent_loss.reshape(-1))
            losses[1].append(pact_loss.reshape(-1))
            ents[0].append(intent_ent.reshape(-1))
            ents[1].append(pact_ent.reshape(-1))
            values.append(value.view(-1))

        # regular = torch.cat(penalties)
        regular = None

        value_loss = []
        pg_losses = ([], [])
        ret = torch.tensor([], device=values[0].device, dtype=torch.float)
        cur_size = 0
        # td_error = adv = (discount*v[s']+r - v[s])
        for i in range(len(batch_iter)-1, -1, -1):
            # mid reward = 0,
            # discount*v[s']+r = discount*v[s']
            if ret.shape[0] > 0:
                ret = discount * values[i+1][:ret.shape[0]].detach()

            # s' do not exist
            # discount*v[s']+r = r
            if cur_size < batch_iter[i].size:
                step = batch_iter[i].size - cur_size
                tmp = torch.tensor(reward[cur_size:cur_size+step], device=value[0].device, dtype=torch.float)
                ret = torch.cat([ret, tmp])
                cur_size += step
            # value loss
            value_loss.append(F.mse_loss(values[i], ret, reduction='none'))
            # self._compute_loss(None, value=values[i], oracle=ret[:cur_size], loss=self.critic_loss)
            # policy loss
            adv = ret-values[i].detach()
            # print('infos', ret[:cur_size].shape, values[i].shape, losses[0][i].shape, losses[1][i].shape, adv.shape)
            pg_losses[0].append(adv*losses[0][i])
            pg_losses[1].append(adv*losses[1][i])

        value_loss = torch.cat(value_loss, dim=0)
        pg_losses = tuple(torch.cat(pl, dim=0) for pl in pg_losses)
        ents = tuple(torch.cat(e, dim=0) for e in ents)
        losses = tuple(torch.cat(e, dim=0) for e in losses)

        return pg_losses, ents, value_loss, regular, (losses, policy_stats)

    def update_a2c(self, args, batch_iters, rewards, model, critic, discount=1, update_table=None):
        if update_table is None:
            update_table = {'value': False, 'policy': False}
        pg_losses, e_losses, value_loss, p_losses = None, None, None, None
        policy_stats = Statistics()
        cur = 0
        for i, bi in enumerate(batch_iters):
            p, e, v, _, info = self._gradient_accumulation(bi, rewards[cur: cur+bi[0].size], model, critic, discount)
            if pg_losses is None:
                pg_losses, e_losses, value_loss = p, e, v
                p_losses = info[0]
            else:
                pg_losses = tuple(torch.cat([pg_losses[i], p[i]], dim=-1) for i in range(2))
                e_losses = tuple(torch.cat([e_losses[i], e[i]], dim=-1) for i in range(2))
                value_loss = torch.cat([value_loss, v], dim=-1)
                p_losses = tuple(torch.cat([p_losses[i], info[0][i]], dim=-1) for i in range(2))
            policy_stats.update(info[1])

        # Update step
        # p_losses = p_losses.mean()
        # e_losses = e_losses.mean()
        value_loss = value_loss.mean()

        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        # print('pgl', pg_losses[0])
        # print('el', e_losses[0])
        model_loss = tuple(pg_losses[i].mean() - self.ent_coef * e_losses[i].mean() for i in range(2))
        critic_loss = self.val_coef * value_loss
        pg_loss = model_loss[0] + model_loss[1]
        total_loss = pg_loss + critic_loss

        # print('all loss', final_loss, p_losses, e_losses, value_loss)
        nan_str = "nan: {}, {}\n{}, {}\n{}\n{}, {}".\
            format(torch.isnan(pg_losses[0].mean()), torch.isnan(pg_losses[1].mean()),
                   torch.isnan(e_losses[0].mean()), torch.isnan(e_losses[1].mean()),
                   torch.isnan(value_loss.mean()),
                   torch.isnan(p_losses[0].mean()), torch.isnan(p_losses[1].mean()))
        assert not torch.isnan(total_loss), nan_str
        # final_loss.backward()
        # model_loss.backward()
        # critic_loss.backward()
        # nn.utils.clip_grad_norm(critic.parameters(), 1.)
        # nn.utils.clip_grad_norm(model.parameters(), 1.)
        # self.optim.step()

        # if not self.model_type == "reinforce":
        if not args.only_run:
            if update_table['value']:
                critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                self.optim['critic'].step()

            if update_table['policy']:
                model.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                self.optim['model'].step()

        ret = {'total_loss': total_loss,
               'pg_loss': pg_loss,
               'pg_loss0': model_loss[0],
               'pg_loss1': model_loss[1],
               'value_loss': critic_loss,
               'entropy0': e_losses[0],
               'entropy1': e_losses[1],
               'policy_loss0': p_losses[0],
               'policy_loss1': p_losses[1],
        }
        return {k: ret[k].reshape(1, -1).cpu().data.numpy() for k in ret}

    def validate(self, args, valid_size, valid_critic=False, start=0, split='dev', exchange=None):
        rate = 0.5
        if exchange is not None:
            if exchange:
                rate = 1
            else:
                rate = 0
        self.model.eval()
        self.critic.eval()
        total_stats = RLStatistics()
        oppo_total_stats = RLStatistics()
        valid_size = min(valid_size, 200)
        # print('='*20, 'VALIDATION', '='*20)
        examples = []
        verbose_str = []
        for sid, scenario in enumerate(self.scenarios[split][start:start+valid_size]):
            controller = self._get_controller(scenario, split=split, rate=rate)
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            rewards = [self.get_reward(example, controller.sessions[i]) for i in range(2)]
            stats = RLStatistics(reward=rewards[0], n_words=1)
            oppo_stats = RLStatistics(reward=rewards[1], n_words=1)
            total_stats.update(stats)
            oppo_total_stats.update(oppo_stats)
            examples.append(example)
            stra = [controller.sessions[i].price_strategy for i in range(2)]
            verbose_str.append(self.example_to_str(example, controller, rewards, sid+start, stra))
        # print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        self.critic.train()
        return [total_stats, oppo_total_stats], examples, verbose_str

    def save_best_checkpoint(self, checkpoint, opt, valid_stats, score_type='accu'):

        if self.best_valid_reward is None:
            better = True
        else:
            if score_type == 'accu' or score_type == 'reward':
                better = valid_stats > self.best_valid_reward
            else:
                better = valid_stats < self.best_valid_reward

        if better:
            path = '{root}/{model}_best.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename)
            print('[Info] Update new best model({}:{:.4f}) at {}.'.format(score_type, valid_stats, path))
            self.best_valid_reward = valid_stats
        # if path is not None:
        #     print('[Info] Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats, score_type='loss'):
        path = '{root}/{model}_{score_type}{score:.4f}_e{episode:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    score_type=score_type,
                    score=stats,
                    episode=episode)
        assert path is not None
        return path

    def update_opponent(self, type=None):
        if type is None:
            types = ['policy', 'critic']
        elif not isinstance(type, list):
            types = [type]
        else:
            types = type

        print('update opponent model for {}.'.format(types))
        if 'policy' in types:
            tmp_model_dict = self.agents[self.training_agent].env.model.state_dict()
            self.agents[self.training_agent^1].env.model.load_state_dict(tmp_model_dict)
        if 'critic' in types:
            tmp_model_dict = self.agents[self.training_agent].env.critic.state_dict()
            self.agents[self.training_agent^1].env.critic.load_state_dict(tmp_model_dict)

    def get_temperature(self, epoch, batch_size, args):
        # deprecated
        return 1
        if args.only_run or args.warmup_epochs == 0:
            return 1
        half = args.num_dialogues // batch_size / 2
        t_s, t_e = 0.3, 1
        i_s, i_e = 0, half
        return min(t_e, t_s + (t_e - t_s) * 1. * epoch / args.warmup_epochs)
        # return min(1., 1.*epoch/half)

    @staticmethod
    def merge_policy(i_policy, p_policy):
        actions = LFSampler._rl_actions
        policy = torch.zeros(len(actions), dtype=torch.float32)
        # print('merge, ', i_policy, p_policy)
        i_policy = i_policy.reshape(-1)
        p_policy = p_policy.reshape(-1)

        for i, act in enumerate(actions):
            policy[i] = i_policy[act[0]]
            if act[1] is not None:
                policy[i] = policy[i] * p_policy[act[1]]

        return policy

    # @staticmethod
    def sort_policy(self, policy, actions, display_num=-1, to_word=str):
        # print('sort', policy, actions)
        scored_actions = [(policy.reshape(-1)[i].data.item(), actions[i]) for i in range(len(actions))]
        scored_actions = sorted(scored_actions, reverse=True, key=lambda x: x[0])
        if display_num == -1:
            return scored_actions
        s = ""
        for i in range(display_num):

            sp, sa = scored_actions[i]
            if isinstance(sa, tuple):
                act = self.lf_vocab.to_word(sa[0])
                if sa[1] is not None:
                    act = act + "," + str(sa[1])
            else:
                act = to_word(sa)
            s = s + "{}:{:.3f} ".format(act, sp)
        return scored_actions, s

    def append_policy_info(self, e, ret, prefix="", display_num=3):
        output_data = e.metadata['output_data']
        pact_size = np.prod(output_data['p_policy'].shape)
        use_tom = output_data.get('tominf_p') is not None

        # print(LFSampler.INTENT_NUM, LFSampler._rl_actions)

        # sl agent
        if pact_size == 1:
            _, s = self.sort_policy(output_data['policy'], list(range(LFSampler.INTENT_NUM)),
                                    display_num, self.lf_vocab.to_word)
            ret.append(prefix+"policy: "+s)
        else:
            # rl agent
            _, s = self.sort_policy(output_data['policy'], list(range(LFSampler.INTENT_NUM)),
                                    display_num, self.lf_vocab.to_word)
            ret.append(prefix+"i_policy: "+s)
            _, s = self.sort_policy(output_data['p_policy'], list(range(LFSampler.PACT_NUM)), display_num)
            ret.append(prefix+"p_policy: "+s)
            policy = RLTrainer.merge_policy(output_data['policy'], output_data['p_policy'])
            _, s = self.sort_policy(policy, LFSampler._rl_actions, display_num*2)
            ret.append(prefix + "policy: " + s)

            # tom agent
            if use_tom:
                _, s = self.sort_policy(output_data['tominf_p2'], LFSampler._rl_actions, display_num*2)
                ret.append(prefix + "tom_p2: " + s)
                _, s = self.sort_policy(output_data['tominf_p'], LFSampler._rl_actions, display_num * 2)
                ret.append(prefix + "tom_p: " + s)
                _, s = self.sort_policy(output_data['tom_ev'], LFSampler._rl_actions, display_num * 2)
                ret.append(prefix + "tom_ev: " + s)

    def example_to_text(self, example):
        ret = []
        for i, e in enumerate(example.events):
            if "real_uttr" in e.metadata.keys():
                ret.append("[{}: {}]\t{}\t{}\t\"{}\"".format(e.time, e.agent, e.action, e.data, e.metadata["real_uttr"]))
            else:
                intent = e.metadata.get('intent')
                intent = self.lf_vocab.to_word(intent)
                ret.append("[{}: {}]\t{}\t{}".format(e.time, e.agent, e.action, e.data))
                ret.append("        <{}>\t{}\t{}".format(intent, e.metadata.get('price'), e.metadata.get('price_act')))
                self.append_policy_info(e, ret, "  ")
                # ret.append("        <{}>\t{}\t{}".format(, e.metadata.get('price'), e.metadata.get('price_act')))
        return ret 

    def example_to_str(self, example, controller, rewards, sid=None, strategies=None):
        if strategies is None:
            strategies = [None, None]
        verbose_str = []
        from core.price_tracker import PriceScaler
        if sid is not None:
            verbose_str.append('[Scenario id: {}]'.format(sid))
        for session_id, session in enumerate(controller.sessions):
            bottom, top = PriceScaler.get_price_range(session.kb)
            s = 'Agent[{}: {}], bottom ${}, top ${}'.format(session_id, session.kb.role, bottom, top)
            verbose_str.append(s)
        verbose_str.append("They are negotiating for "+session.kb.facts['item']['Category'])
        verbose_str.append("strategy: {}, {}".format(strategies[0], strategies[1]))

        strs = self.example_to_text(example)
        for str in strs:
            verbose_str.append(str)
        s = "reward: [0]{}\nreward: [1]{}".format(rewards[0], rewards[1])
        verbose_str.append(s)
        return verbose_str

    def get_eval_dict(self, examples, strategies):
        eval_dict, separate_edict = {}, [{} for _ in range(10)]
        # len, s_rate, utility, fairness
        for i, e in enumerate(examples):
            role = e.scenario.kbs[0].role
            l = len(e.events)
            srate = self._is_agreed(e)
            reward = self._base_utility(e)[role]
            ut = max(reward, 0)
            fa = 1 - 2 * abs(ut - 0.5)
            tmp_dict = {'length': l, 'success_rate': srate, 'reward': reward}
            tmp_dict['utility'] = ut
            tmp_dict['fairness'] = fa
            for k in tmp_dict:
                if eval_dict.get(k) is None:
                    eval_dict[k] = []
                if separate_edict[strategies[i]].get(k) is None:
                    separate_edict[strategies[i]][k] = []
                eval_dict[k].append(tmp_dict[k])
                separate_edict[strategies[i]][k].append(tmp_dict[k])

        return eval_dict, separate_edict

    def sample_data(self, i, sample_size, args, real_batch=None, batch_size=128, eval=False):
        if real_batch is None:
            real_batch = sample_size
        rewards = [0]*2
        s_rewards = [0]*2
        _batch_iters = [[], []]
        _rewards = [[], []]
        examples = []
        verbose_strs = []
        strategies = [[], []]

        dialogue_batch = [[], []]
        last_t = time.time()
        for j in range(real_batch):
            # Rollout
            if eval:
                scenario, sid = self._get_scenario(scenario_id=j)
                controller = self._get_controller(scenario, split='train', rate=0)
            else:
                scenario, sid = self._get_scenario()
                controller = self._get_controller(scenario, split='train')
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose, temperature=self.get_temperature(i, sample_size, args))

            for session_id, session in enumerate(controller.sessions):
                # if args.only_run != True and session_id != self.training_agent:
                #     continue
                # Compute reward
                reward = self.get_reward(example, session)
                # Standardize the reward
                all_rewards = self.all_rewards[session_id]
                all_rewards.append(reward)
                s_reward = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))

                rewards[session_id] = reward
                s_rewards[session_id] = s_reward
                _rewards[session_id].append(reward)
                strategies[session_id].append(session.price_strategy_label)

            for session_id, session in enumerate(controller.sessions):
                # dialogue_batch[session_id].append(session.dialogue)
                # if len(dialogue_batch[session_id]) == batch_size or j == real_batch-1:
                batch_iter = session.iter_batches()
                T = next(batch_iter)
                _batch_iters[session_id].append(list(batch_iter))

            stra = [controller.sessions[i].price_strategy for i in range(2)]
            examples.append(example)
            verbose_str = self.example_to_str(example, controller, rewards, sid,
                                              stra)

            if args.verbose:
                for s in verbose_str:
                    print(s)
            verbose_strs.append(verbose_str)

            # print('t: ', time.time() - last_t)
            # last_t=time.time()

        return _batch_iters, (_rewards, strategies), examples, verbose_strs

    def learn(self, args):
        rewards = [None]*2
        s_rewards = [None]*2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        save_every = 100

        history_train_losses = [[],[]]

        batch_size = 100

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        for i in range(args.num_dialogues // batch_size):
            _batch_iters, _rewards, example, train_ex_str = self.sample_data(i, batch_size, args)
            # print('reward is:', _rewards)
            # print(np.mean(_rewards[0]), np.mean(_rewards[1]))
            # print(np.mean(self.all_rewards[0][-tensorboard_every*batch_size:]), np.mean(self.all_rewards[1][-tensorboard_every*batch_size:]))

            path_txt = '{root}/{model}_example{epoch}.txt'.format(
                root=args.model_path,
                model=args.name,
                epoch=i)
            with open(path_txt, 'w') as f:
                for ex in train_ex_str:
                    f.write('-' * 7 + '\n')
                    for s in ex:
                        f.write(s + '\n')

                # if train_policy:
                #     self.update(batch_iter, reward, self.model, discount=args.discount_factor)
                #
                # if train_critic:
                #     stats = self.update_critic(batch_iter, reward, self.critic, discount=args.discount_factor)
                #     critic_report_stats.update(stats)
                #     critic_stats.update(stats)
            k = -1
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
                # if (k+1)%5 == 0:
                #     _batch_iters, _rewards, example, _ = self.sample_data(i, batch_size, args)
                # if loss[0,3].item() < 0.2:
                #     break
            if k >=0:
                print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k+1, loss[0,3].item()))
            # if loss[0, 3].item() >= 0.3:
            #     print('Try to initialize critic parameters.')
            #     for p in self.critic.parameters():
            #         p.data.uniform_(-args.param_init, args.param_init)
            #     for k in range(20):
            #         loss = self.update_a2c(args, _batch_iters, _rewards, self.model, self.critic,
            #                                discount=args.discount_factor, fix_policy=True)
            #         if (k + 1) % 5 == 0:
            #             _batch_iters, _rewards, controller, example = self.sample_data(i, batch_size, args)
            #         if loss[0, 3].item() < 0.2:
            #             break
            #     print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k + 1,
            #                                                                                       loss[0, 3].item()))
            loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                   discount=args.discount_factor)
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
            history_train_losses[self.training_agent].append(loss)

            # print('verbose: ', args.verbose)

                    # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            # Save logs on tensorboard
            if (i + 1) % tensorboard_every == 0:
                ii = (i+1)*batch_size
                for j in range(2):
                    self.writer.add_scalar('agent{}/reward'.format(j), np.mean(self.all_rewards[j][-tensorboard_every*batch_size:]), ii)
                    if len(history_train_losses[j]) >= tensorboard_every*batch_size:
                        tmp = np.concatenate(history_train_losses[j][-tensorboard_every*batch_size:], axis=0)
                        tmp = np.mean(tmp, axis=0)
                        self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], ii)
                        self.writer.add_scalar('agent{}/policy_loss'.format(j), tmp[1], ii)
                        self.writer.add_scalar('agent{}/entropy_loss'.format(j), tmp[2], ii)
                        self.writer.add_scalar('agent{}/value_loss'.format(j), tmp[3], ii)
                        self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[4], ii)
                        self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[5], ii)
                        self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[6], ii)


            if ((i + 1) % report_every) == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt
                if args.histogram:
                    sns.set_style('darkgrid')

                # if train_policy:
                for j in range(2):
                    print('agent={}'.format(j), end=' ')
                    print('step:', i, end=' ')
                    print('reward:', rewards[j], end=' ')
                    print('scaled reward:', s_rewards[j], end=' ')
                    print('mean reward:', np.mean(self.all_rewards[j][-args.report_every:]))
                    if args.histogram:
                        self.agents[j].env.dialogue_generator.get_policyHistogram()

                # if train_critic:
                #     critic_report_stats.output(i+1, 0, 0, last_time)
                #     critic_report_stats = RLStatistics()

                print('-'*10)
                if args.histogram:
                    plt.show()

                last_time = time.time()

            # Save model
            if (i+1) % save_every == 0:
                # TODO: valid in dev set
                valid_stats, _, _ = self.validate(args, 50 if args.only_run else 200)
                valid_stats = valid_stats[0]
                if not args.only_run:
                    self.drop_checkpoint(args, i+1, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                    if args.update_oppo:
                        print('update oppo!')
                        self.update_opponent(['policy', 'critic'])
                else:
                    print('valid ', valid_stats.str_loss())

                # if train_policy:
                #     valid_stats, _ = self.validate(args)
                #     self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     self.update_opponent('policy')
                #
                # elif train_critic:
                #     # TODO: reverse!
                #     self.drop_checkpoint(args, i, critic_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     critic_stats = RLStatistics()
                # else:
                #     valid_stats, _ = self.validate(args)
                #     print('valid result: ', valid_stats.str_loss())