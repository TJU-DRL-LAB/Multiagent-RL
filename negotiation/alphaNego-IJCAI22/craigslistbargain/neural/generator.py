import copy
import random

import torch
from torch.autograd import Variable
import onmt.io
from onmt.Utils import aeq, use_gpu

from cocoa.core.entity import is_entity
from cocoa.neural.generator import Generator, Sampler

from .symbols import markers, category_markers, sequence_markers
from .utterance import UtteranceBuilder
from .dsac_utils import *
from cocoa.core.entity import is_entity, Entity, CanonicalEntity

import numpy as np


class LFSampler(Sampler):

    var_for_price = 0.1
    INTENT_NUM = -1

    def __init__(self, model, vocab,
                 temperature=1, max_length=100, cuda=False, model_type='sl'):
        super(LFSampler, self).__init__(model, vocab, temperature=temperature, max_length=max_length, cuda=cuda)
        self.acc_or_rej = list(map(self.vocab.to_ind, ('accept', 'reject')))
        self.offer = list(map(self.vocab.to_ind, ('offer', )))
        self.price_actions = list(map(self.vocab.to_ind, ('counter', 'propose', 'offer', 'agree', 'disagree')))
        self.model_type = model_type

        self.policy_history = []

        # init rl actions
        _ = self.rl_actions
        LFSampler.INTENT_NUM = len(self.vocab.ind_to_word)



    @staticmethod
    def softmax_with_mask(policy, mask=1, eval=False):
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=policy.device)
        origin_policy = torch.clone(policy)
        if len(policy.size()) == 1:
            policy = policy.unsqueeze(dim=0)
        policy.sub_(policy.max(1, keepdim=True)[0].expand(-1, policy.size(1)))
        # mask = batch.policy_mask
        # policy[mask == 0] = -100.
        # print(batch.policy_mask)

        # Avoid policy equal to zero ( + 1e-6 )
        p_exp = (policy.exp() + 1e-6).mul(mask)
        # p_exp = (policy.exp() + 1e-6)
        # if torch.sum(p_exp).item() == 0:
        #     p_exp += torch.ones_like(p_exp).mul(batch.policy_mask)

        policy = p_exp / (torch.sum(p_exp, keepdim=True, dim=1))
        if torch.any(policy < 0) or torch.any(torch.isnan(policy)):
            print('lots of errors: ', p_exp, mask)

        act = torch.multinomial(policy, 1).reshape(-1)
        # if eval:
        #     act = policy.argmax()
        # (batch_size,), (batch_size, intent_size)
        prob = random.random()

        return act, policy

    def _run_batch_tom_identity(self, batch, hidden_state, only_identity=False, id_gt=False):
        if id_gt:
            id_gt = batch.strategy
        else:
            id_gt = None
        if only_identity:
            identity, next_hidden = \
                self.model.encoder.identity(batch.identity_state, batch.extra, hidden_state, uttr=batch.uttr)
            predictions = None
        else:
            output = self.model(batch.uttr, batch.identity_state, batch.state,
                              batch.extra, hidden_state, id_gt)
            if len(output) == 3:
                predictions, next_hidden, identity = output
            else:
                predictions, next_hidden = output
                identity = None
        return predictions, next_hidden, identity

    def generate_batch(self, batch, gt_prefix=1, enc_state=None, whole_policy=False, special_actions=None,
                       temperature=1, acpt_range=None, hidden_state=None, eval=False):
        # This is to ensure we can stop at EOS for stateful models
        assert batch.size == 1.

        # Run the model
        if self.model_type in ['sl', 'rl', 'pt-neural-dsac']:
            policy, p_policy = self.model(batch.uttr, batch.state)
            next_hidden = None
        else:
            # tom model
            out, next_hidden, _ = self._run_batch_tom_identity(batch, hidden_state)
            policy, p_policy = out

        last_intent, last_price = batch.get_pre_info(self.vocab)
        # if self.model_type == "pt-neural-dsac" and torch.any(torch.isnan(policy)):
        #     print("error for nan origin policy", policy)
        #     print("uttr and state")
        #     print(batch.uttr, batch.state)
        p_mask = batch.policy_mask(self.vocab)

        if self.model_type == 'pt-neural-dsac':
            intent, policy = self.softmax_with_mask(policy, p_mask, eval=eval)
        else:
            intent, policy = self.softmax_with_mask(policy, p_mask)

#{'offer': 6, 'disagree': 17, 'counter': 2, '<unknown>': 1, 'reject': 19, 'counter-noprice': 5, 'None': 4,
# #'inquire': 9, 'greet': 3, 'deny': 14, 'confirm': 16, 'affirm': 10,
# #'agree-noprice': 7, 'inform': 13, 'agree': 15, 'accept': 8, 'start': 12, 'propose': 11, 'quit': 18, '<pad>': 0}
        prob = policy[0, intent.item()].item()
        price = 1
        if self.model_type == "sl" and  acpt_range is None:
            intent = input()
            intent = int(intent)
        elif self.model_type == 'rl':
            p_act, p_policy = self.softmax_with_mask(p_policy)
            price = p_act.item()
            prob = prob*p_policy[0, price].item()
        elif self.model_type == 'pt-neural-dsac':
            p_act, p_policy = self.softmax_with_mask(p_policy, eval=eval)

            price = p_act.item()
            prob = prob*p_policy[0, price].item()
            def _pact_to_price(p_act, p_last):
                pmax, pmin = p_last[0], p_last[1]
                p = 1
                if p_act == 0:
                    # insist
                    p = pmax
                elif p_act == 1:
                    p = pmin

                elif p_act == 2:
                    p = (pmax + pmin) / 2
                elif p_act == 3:
                    p = pmax - 0.1
                else:
                    print('what\'s wrong?', p_act)

                p = min(p, pmax)
                p = max(p, pmin)
                return p


            if last_intent[0] in self.offer:
                last_prices = [last_price[0, 0].item(), last_price[0, 1].item()]
                offer_price = last_price[0, 1].item()
                current_price = _pact_to_price(price, last_prices)
                if current_price - 1e-6 <= offer_price or offer_price >= 0.25:
                # if current_price - 1e-6 <= offer_price:
                # if offer_price >= 0.4:
                    act_idx = self.acc_or_rej[0]
                else:
                    act_idx = self.acc_or_rej[1]
                intent = act_idx

        else:
            price = p_policy.item()

        # Use Normal distribution with constant variance as policy on price

        # Use rule for Supervised learning agent
        if acpt_range is not None:
            assert len(acpt_range) == 2
            # Check if last action is offer
            if last_intent[0] in self.offer:
                offer_price = last_price[0, 1].item()
                policy = policy * 0

                if (acpt_range[0] >= offer_price) and (offer_price >= acpt_range[1]):
                    # Accept
                    act_idx = self.acc_or_rej[0]

                else:
                    act_idx = self.acc_or_rej[1]

                intent[0] = act_idx
                policy[0, act_idx] = 1
                prob = 1.

        # TODO: Not correct, for multiple data.
        original_price = price
        if intent not in self.price_actions:
            price = None


        # print('gen output: ',policy, price)
        ret = {"intent": intent,
               "price": price,
               "policy": policy,
               "p_policy": p_policy,
               'prob': prob,
               'original_price': original_price,
               'rnn_hidden': next_hidden
               }
        # ret["batch"] = batch
        return ret

    def get_sl_actions(self, mean, logstd, sample_num=1):
        # get all actions
        all_actions = []
        num_actions = len(self.vocab.word_to_ind)
        actions = list(range(num_actions))
        for i in actions:
            if not i in self.price_actions:
                all_actions.append([i, None])
            else:
                for j in range(sample_num):
                    all_actions.append([i, mean])
        return all_actions

    _rl_actions = None
    _rl_act_index = None
    PACT_NUM = 4

    @property
    def rl_act_index(self):
        if self._rl_act_index is None:
            _ = self.rl_actions
        return self._rl_act_index

    @property
    def rl_actions(self):
        if LFSampler._rl_actions is not None:
            return LFSampler._rl_actions

        # get all actions
        all_actions = []
        num_actions = len(self.vocab.word_to_ind)
        actions = list(range(num_actions))
        idx = 0
        LFSampler._rl_actions = {}
        for i in actions:
            if not i in self.price_actions:
                all_actions.append((i, None))
                LFSampler._rl_actions[all_actions[-1]] = idx
                idx += 1
            else:
                for j in range(self.PACT_NUM):
                    all_actions.append((i, j))
                    LFSampler._rl_actions[all_actions[-1]] = idx
                    idx += 1

        LFSampler._rl_actions = all_actions
        return all_actions

    def _old_get_all_actions(self, mean, logstd, p_num=5, no_sample=False):
        # prices
        self.prices = []
        step = 0.05
        now = step
        while now < 1:
            self.prices.append(now)
            now += step
        num_acions = len(self.vocab.word_to_ind)
        self.actions = list(range(num_acions))

        # get all actions
        all_actions = []
        for i in self.actions:
            if not i in self.price_actions:
                # print('a ', self.vocab.to_word(i))
                all_actions.append((i,None))
        d = torch.distributions.normal.Normal(mean, logstd.exp())

        for i in self.price_actions:
            # print('pa ', self.vocab.to_word(i))
            if no_sample:
                now = 0.
                step = 0.01
                p_list = []
                while now <= 1:
                    p_list.append(now)
                    now += step
                prob = d.log_prob(torch.from_numpy(np.array(p_list)).float().to(mean.device)).exp()
                prob = prob / prob.sum(dim=-1, keepdim= True)
                for j, p in enumerate(p_list):
                    pp = prob[j].cpu().item()
                    if pp >= 1e-6:
                        all_actions.append((i,p, pp))

            else:
                for j in range(p_num):
                    p = mean + logstd.exp() * torch.randn_like(mean)
                    p = p.cpu().item()
                    all_actions.append((i,p))
        return all_actions

    def get_policyHistogram(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import re

        allNum = len(self.policy_history)
        tmpData = np.mean(self.policy_history,axis=0)[0]
        r = re.compile(u'\d+[.]?\d*')
        x, w = [], []
        for i in range(len(tmpData)):
            tmp = self.vocab.ind_to_word[i]
            if not is_entity(tmp):
                continue
            name = tmp.canonical.value
            if abs(name) > 10.1: continue
            x.append(name)
            w.append(tmpData[i])

        w = w/np.sum(w)
        from scipy.stats import norm
        sns.distplot(x, bins=100, kde=False, hist_kws={'weights': w}, )
        #plt.show()



def get_generator(model, vocab, scorer, args, model_args):
    from onmt.Utils import use_gpu
    if args.sample:
        if model_args.model == 'lf2lf':
            generator = LFSampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
        else:
            generator = Sampler(model, vocab, args.temperature,
                                max_length=args.max_length,
                                cuda=use_gpu(args))
    else:
        generator = Generator(model, vocab,
                              beam_size=args.beam_size,
                              n_best=args.n_best,
                              max_length=args.max_length,
                              global_scorer=scorer,
                              cuda=use_gpu(args),
                              min_length=args.min_length)
    return generator
