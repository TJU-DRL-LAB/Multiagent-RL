import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict
from .risk import distortion_de
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.adam import Adam
from core.controller import Controller
from .utterance import UtteranceBuilder
from core.scenario import Scenario
from systems import get_system
from tensorboardX import SummaryWriter
import pickle as pkl
from neural.batcher_rl import RLBatch, RawBatch, ToMBatch
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from neural.rl_trainer import RLTrainer as BaseTrainer
from neural.sl_trainer import Statistics, SimpleLoss
from neural.generator import LFSampler
from .dsac_utils import *
import math, time, sys
import onmt.pytorch_utils as ptu
from cocoa.core.util import read_json


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


def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    device = input.device
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    sign = sign.to(device)
    L = L.to(device)
    weight = weight.to(device)
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()

class DSACTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin',
                 cuda=False, args=None):

        super(DSACTrainer, self).__init__(agents, scenarios, train_loss, optim,
                                        training_agent, reward_func, cuda, args)
        # print('training_agent', training_agent)
        self.dsac = agents[training_agent].env.dsac
        self.sl_policy = self.dsac.sl_model
        self.policy = self.dsac.actor_model
        self.target_policy = self.dsac.target_actor_model
        self.zf1 = self.dsac.critic_model1
        self.zf2 = self.dsac.critic_model2
        self.target_zf1 = self.dsac.target_critic_model1
        self.target_zf2 = self.dsac.target_critic_model2
        self.soft_target_tau = 5e-3
        self.target_update_period = 1
        self.tau_type = 'iqn'
        self.zf_criterion = quantile_regression_loss
        self.num_quantiles = 8
        self.fp = None
        self.target_fp = None
        self.vocab = agents[training_agent].env.vocab
        self.lf_vocab = agents[training_agent].env.lf_vocab
        #'counter', 'propose', 'offer', 'agree', 'disagree'
        # {'deny': 14, 'reject': 19, '<pad>': 0, 'disagree': 17, 'confirm': 16, 'accept': 8, 'counter': 2, 'None': 4,
        #  'quit': 18, 'affirm': 10, 'inquire': 9, 'counter-noprice': 5, 'agree': 15, 'inform': 13, 'start': 12,
        #  'offer': 6, 'greet': 3, 'propose': 11, 'agree-noprice': 7, '<unknown>': 1}
        self.price_action_indexs = [2,11,6,15,17]
        self._price_actions_masks = torch.sum(F.one_hot(torch.LongTensor(self.price_action_indexs), num_classes=self.dsac.intent_size), dim=0).float()
        self.model_type = args.model_type
        self.use_utterance = False
        self.hidden_vec = None
        self.target_tau = 'iqn'
        self.reward_scale = 1.0
        self.risk_type = 'neutral'
        self.discount = 0.99
        self.zf_lr =  args.zf_lr
        self.policy_optimizer = Adam([{'params':self.policy.decoder.common_net.parameters(),'lr':args.common_lr},
                                      {'params':self.policy.decoder.intent_net.parameters(),'lr':args.intent_lr},
                                      {'params':self.policy.decoder.price_net.parameters(),'lr':args.price_lr}])
        self.zf1_optimizer = Adam(self.zf1.parameters(), lr=self.zf_lr)
        self.zf2_optimizer = Adam(self.zf2.parameters(), lr=self.zf_lr)
        risk_param_final = None
        risk_schedule_timesteps=1
        risk_param=0
        self.risk_schedule = LinearSchedule(risk_schedule_timesteps, risk_param,
                                            risk_param if risk_param_final is None else risk_param_final)
        self._n_train_steps_total = 0
        self.clip_norm = 0.0
        self.alpha = args.alpha
        self.args = args
        if args.self_play:
            self.opponents_pool = {"sl":{},"rl":{}, "self":{}}
            self.score_table={"sl":{}, "rl":{}, "self":{}}
            self.rule_strategies = ['insist', 'decay', 'persuaded', 'convex', 'concave', 'low', 'high', 'behavior_kind', 'behavior_unkind']
            schema = Schema(args.schema_path)
            scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
            valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)
            for i in self.rule_strategies:
                tmp_system = get_system("pt-neural-r", args, schema, False, "checkpoint/language/model_best.pt", id=1)
                tmp_system.env = tmp_system.env._replace(price_strategy=i)
                self.opponents_pool['sl'][i] = tmp_system
                self.score_table['sl'][i] = {'success_rate':[0.0],'utility':[0.0],'length':[10],'fairness':[0],'score':[1], 'win_rate':[0]}
            self.score_table['self']['self_agent'] = {'success_rate':[0.0],'utility':[0.0],'length':[10],'fairness':[0],'score':[1], 'win_rate':[0]}


    def calculate_score(self, ag, ut, l, fair):
        print("fair", fair)
        assert 0 <= ag and ag <= 1
        assert 0 < l and l <= self.args.max_length
        alpha = -5e-3
        beta = 0.1
        score = math.pow(1 - ag + 0.1, -ut) + alpha * l + beta * fair
        return score

    def update_score_table(self, info):
        opponent_type = self.score_table[info['opponent_type']]
        if opponent_type.get(info['opponent_name']) is None:
            opponent_type[info['opponent_name']] = {'success_rate': [0.0], 'utility': [0.0], 'length': [10], 'fairness': [0.0], 'score': [1.0], 'win_rate':[0.0]}
        item = opponent_type[info['opponent_name']]
        item['success_rate'].append(info['success_rate'])
        item['utility'].append(info['utility'])
        item['length'].append(info['length'])
        item['fairness'].append(info['fairness'])
        item['score'].append(self.calculate_score(info['success_rate'],info['utility'],info['length'],info['fairness']))
        item['win_rate'].append(info['win_rate'])



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
            example = controller.simulate(args.max_turns, verbose=args.verbose, temperature=1, eval=eval)

            for session_id, session in enumerate(controller.sessions):
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

        return _batch_iters, (_rewards, strategies), examples, verbose_strs

    def get_opponent_system(self):
        if len(self.opponents_pool['rl']) == 0:
            sample_kind = np.random.choice(['sl', 'self'], p=[0,1])
        else:

            sample_kind = np.random.choice(['sl', 'rl', 'self'], p=[0,0,1])
        if sample_kind in ['sl','rl']:

            prob = [self.score_table[sample_kind][item]['win_rate'][-1] for item in self.score_table[sample_kind].keys()]
            prob = self.calculate_pfsp_prob(prob)
            # print(prob)
            specific_opponent = np.random.choice(list(self.opponents_pool[sample_kind].keys()), p=prob)
            opponent = self.opponents_pool[sample_kind][specific_opponent]
        else:
            opponent = self.agents[0]
        return opponent

    def calculate_pfsp_prob(self, prob):
        prob = np.array(prob)
        prob1 = [1 / (x + 0.0001) for x in prob]
        prob1 = np.array(prob1)
        prob2 = prob1 / sum(prob1)
        return prob2

    def sample_data_pfsp(self, i, sample_size, args, real_batch=None, batch_size=128, eval=False):
        if real_batch is None:
            real_batch = sample_size
        rewards = [0]*2
        s_rewards = [0]*2
        _batch_iters = [[], []]
        _rewards = [[], []]
        examples = []
        verbose_strs = []
        strategies = [[], []]

        for j in range(real_batch):
            # Rollout

            opponent = self.get_opponent_system()
            if eval:
                scenario, sid = self._get_scenario(scenario_id=j)
                controller = self._get_controller_(scenario, split='train', rate=0)
            else:
                scenario, sid = self._get_scenario()
                controller = self._get_controller_pfsp(scenario, split='train', opponent=opponent)
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose, temperature=1, eval=eval)

            for session_id, session in enumerate(controller.sessions):
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

        return _batch_iters, (_rewards, strategies), examples, verbose_strs

    def _get_controller_pfsp(self, scenario, opponent, split='train', rate=0.5):
        # Randomize
        if random.random() < rate:
            scenario = copy.deepcopy(scenario)
            scenario.kbs = (scenario.kbs[1], scenario.kbs[0])
        sessions = [self.agents[0].new_session(0, scenario.kbs[0]),
                    opponent.new_session(1, scenario.kbs[1])]
        return Controller(scenario, sessions)

    def example_to_str(self, example, controller, rewards, sid=None, strategies=None):
        if strategies is None:
            strategies = [None, None]
        verbose_str = []
        from core.price_tracker import PriceScaler
        if sid is not None:
            verbose_str.append('[Scenario id: {}]'.format(sid))
        for session_id, session in enumerate(controller.sessions):
            bottom, top = PriceScaler.get_price_range(session.kb)
            s = 'Agent[{}: {}], top ${} bottom ${}'.format(session_id, session.kb.role, top, bottom)
            verbose_str.append(s)
        verbose_str.append("They are negotiating for "+session.kb.facts['item']['Category'])
        verbose_str.append("strategy: {}, {}".format(strategies[0], strategies[1]))

        strs = self.example_to_text(example)
        for str in strs:
            verbose_str.append(str)
        s = "reward: [0]{}\nreward: [1]{}".format(rewards[0], rewards[1])
        verbose_str.append(s)
        return verbose_str

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
            policy = DSACTrainer.merge_policy(output_data['policy'], output_data['p_policy'])
            _, s = self.sort_policy(policy, LFSampler._rl_actions, display_num*2)
            ret.append(prefix + "policy: " + s)

    def get_eval_dict(self, examples, strategies):
        eval_dict, separate_edict = {}, [{} for _ in range(10)]
        # len, s_rate, utility, fairness
        for i, e in enumerate(examples):
            role = e.scenario.kbs[0].role
            l = len(e.events)
            srate = self._is_agreed(e)
            reward = self._margin_reward(e)[role]

            ut = reward / 2 + 0.5
            fa = 1 - abs(reward)
            tmp_dict = {'length': l, 'success_rate': srate, 'reward': reward}
            if srate:
                tmp_dict['utility'] = ut
                tmp_dict['fairness'] = max(fa, 0)
                tmp_dict['fairness'] = min(fa, 1)

            for k in tmp_dict:

                if eval_dict.get(k) is None:
                    eval_dict[k] = []
                if len(strategies) != 0:
                    if separate_edict[strategies[i]].get(k) is None:
                        separate_edict[strategies[i]][k] = []
                    else:
                        separate_edict[strategies[i]][k].append(tmp_dict[k])
                eval_dict[k].append(tmp_dict[k])



        return eval_dict, separate_edict



    def get_opponent_kind(self, opponent):
        if opponent in  self.rule_strategies:
            return "sl"
        elif opponent.startswith("rl_"):
            return "rl"
        elif opponent == "self_agent":
            return "self"

    def get_eval_dict_sp(self, examples, opponent):

        eval_dict = {}

        # len, s_rate, utility, fairness
        cnt = 0
        total_cnt = 0
        win_rate = 0.0
        for i, e in enumerate(examples):
            role = e.scenario.kbs[0].role
            l = len(e.events)
            srate = self._is_agreed(e)
            reward = self._margin_reward(e)[role]
            ut = reward / 2 + 0.5
            fa = 1 - abs(reward)
            tmp_dict = {'length': l, 'success_rate': srate, 'reward': reward}
            if srate:
                tmp_dict['utility'] = ut
                tmp_dict['fairness'] = fa
                tmp_dict['fairness'] = max(tmp_dict['fairness'], 0)
                tmp_dict['fairness'] = min(tmp_dict['fairness'], 1)
                total_cnt += 1
            if reward >= 0:
                cnt += 1


            
            for k in tmp_dict:
                if eval_dict.get(k) is None:
                    eval_dict[k] = []

                eval_dict[k].append(tmp_dict[k])

        avg_success_rate = np.mean(eval_dict['success_rate'])
        avg_utility = np.mean(eval_dict['utility'])
        avg_length = np.mean(eval_dict['length'])
        avg_fairness = np.mean(eval_dict['fairness'])
        num = len(examples)
        print("cnt",cnt)
        print("total_cnt", total_cnt)
        if total_cnt != 0:
            win_rate = 1.0 * cnt / total_cnt
        info = {'success_rate':avg_success_rate,'utility':avg_utility, 'length':avg_length, \
                'fairness':avg_fairness, 'opponent_name':opponent, 'opponent_type':self.get_opponent_kind(opponent), 'win_rate':win_rate}
        self.update_score_table(info)
        return eval_dict

    def get_avg_score(self):
        res = []
        for kind in self.score_table.keys():
            for opponent in self.score_table[kind].keys():
                res.append(self.score_table[kind][opponent]['score'][-1])
        return np.mean(res)

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

    def get_tau(self, actions_size, obs=None,  fp=None, device="cpu"):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(actions_size, self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(actions_size, self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdim=True)

        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau.to(device), tau_hat.to(device), presum_tau.to(device)

    def get_policy_mask(self, intents):
        # if isinstance(intents, torch.Tensor):
        RawBatch.init_vocab(self.vocab)
        policy_mask = np.ones((len(intents), RawBatch.intent_size))

        for i in range(len(intents)):
            if intents[i] == RawBatch.offer_idx:
                policy_mask[i, :] = 0
                policy_mask[i, [RawBatch.acc_idx, RawBatch.rej_idx]] = 1
            elif intents[i] in [RawBatch.acc_idx, RawBatch.rej_idx]:
                policy_mask[i, :] = 0
                policy_mask[i, [RawBatch.quit_idx, ]] = 1
            else:
                policy_mask[i, [RawBatch.acc_idx, RawBatch.rej_idx, RawBatch.pad_idx,
                                RawBatch.none_idx, RawBatch.unk_idx, RawBatch.quit_idx,
                                RawBatch.start_idx]] = 0
        policy_mask = torch.FloatTensor(policy_mask)

        return policy_mask

    def get_price_mask(self, intents, prices):
        pass

    def softmax_with_mask(self, policy, mask=1, eval=False):
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=policy.device)
        policy.sub_(policy.max(1, keepdim=True)[0].expand(-1, policy.size(1)))
        p_exp = (policy.exp() + 1e-6).mul(mask)
        policy = p_exp / (torch.sum(p_exp, keepdim=True, dim=1))
        if torch.any(policy < 0) or torch.any(torch.isnan(policy)):
            print('lots of errors: ', p_exp, mask)
        act = torch.multinomial(policy, 1).reshape(-1)
        return act, policy

    def get_action_full_information(self,model, batch_uttr, batch_state):
        device = batch_state.device
        state_length = (batch_state.size(1) - 5) // (self.dsac.intent_size + 1)
        last_intents_one_hot = batch_state[:,(state_length-1) * self.dsac.intent_size : state_length * self.dsac.intent_size]
        last_intents_index = torch.argmax(last_intents_one_hot, dim=1)
        policy_masks = self.get_policy_mask(last_intents_index).to(device)

        action = model(batch_uttr, batch_state)
        (intent, price) = action

        intent_gumbel_softmax = gumbel_softmax_with_mask(intent,device=device, mask=policy_masks, hard=False)
        price_gumbel_softmax = gumbel_softmax_with_mask(price,device=device,hard=False)
        intent_one_hot = gumbel_softmax_with_mask(intent,device=device,mask=policy_masks, hard=True)
        self._price_actions_masks = self._price_actions_masks.to(device)
        price_mask = torch.sum(intent_one_hot * self._price_actions_masks, dim=1).unsqueeze(dim=1)

        price_one_hot = gumbel_softmax_with_mask(price,device=device,hard=True)
        price_one_hot = price_one_hot * price_mask

        intent_log_probability = torch.log(torch.mul(intent_gumbel_softmax, intent_one_hot) + 1e-10)
        price_log_probability = torch.log(torch.mul(price_gumbel_softmax, price_one_hot) + 1e-10)

        sl_intent_probabilities = self.sl_policy(batch_uttr, batch_state)[0]
        _, sl_intent_softmax = self.softmax_with_mask(sl_intent_probabilities, policy_masks)
        return intent_one_hot, price_one_hot, intent_log_probability.max(dim=1)[0].unsqueeze(dim=1), price_log_probability.max(dim=1)[0].unsqueeze(dim=1), intent_gumbel_softmax, sl_intent_softmax

    def validate(self, args, valid_size, valid_critic=False, start=0, split='dev', exchange=None):
        rate = 0.5
        if exchange is not None:
            if exchange:
                rate = 1
            else:
                rate = 0
        self.dsac.set_eval()

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
            example = controller.simulate(args.max_turns, verbose=args.verbose, eval=True)
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

        self.dsac.set_train()

        return [total_stats, oppo_total_stats], examples, verbose_str


    def validate_sp(self, args, valid_size, opponent, valid_critic=False, start=0, split='dev', exchange=None):
        rate = 0.5
        if exchange is not None:
            if exchange:
                rate = 1
            else:
                rate = 0
        self.dsac.set_eval()
        valid_size = min(valid_size, 200)
        # print('='*20, 'VALIDATION', '='*20)
        examples = []
        verbose_str = []
        for sid, scenario in enumerate(self.scenarios[split][start:start+valid_size]):
            controller = self._get_controller_pfsp(scenario, split=split, rate=0, opponent=opponent)
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose, eval=True)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            rewards = [self.get_reward(example, controller.sessions[i]) for i in range(2)]
            examples.append(example)
            stra = [controller.sessions[i].price_strategy for i in range(2)]
            verbose_str.append(self.example_to_str(example, controller, rewards, sid+start, stra))
        # print('='*20, 'END VALIDATION', '='*20)

        self.dsac.set_train()

        return examples, verbose_str

    def train_from_torch(self, batch, update_policy=True, update_critic=True, is_pretrain=False):
        self.dsac.set_train()
        device = self.dsac.device
        batch_uttr, batch_state, batch_intent, batch_price, batch_price_mask, batch_next_uttr, batch_next_state, batch_reward, batch_done = batch
        batch_size = len(batch_uttr)
        batch_state = torch.stack(batch_state).to(device)
        batch_intent = torch.stack(batch_intent).to(device)
        batch_price = torch.stack(batch_price).to(device)
        batch_price_mask = torch.LongTensor(batch_price_mask).to(device).unsqueeze(dim=1)
        batch_next_state = torch.stack(batch_next_state).to(device)
        batch_reward = torch.FloatTensor(batch_reward).to(device).unsqueeze(dim=1)
        batch_done = torch.FloatTensor(batch_done).to(device).unsqueeze(dim=1)

        intents_onehot = F.one_hot(batch_intent, num_classes=self.dsac.intent_size)
        prices_onehot = F.one_hot(batch_price, num_classes=self.dsac.price_size)
        prices_onehot = prices_onehot * batch_price_mask

        actions = torch.cat((intents_onehot, prices_onehot), dim=1).to(torch.float32)
        new_intents_onehot, new_prices_onehot, new_intents_log_probabilities, new_prices_log_probabilities, \
                                                intent_softmax, sl_intent_softmax = self.get_action_full_information(self.policy, batch_uttr, batch_state)
        new_actions = torch.cat((new_intents_onehot, new_prices_onehot), dim=1).to(torch.float32)

        alpha_loss = 0
        alpha = self.alpha
        zf1_loss = None
        zf2_loss = None
        policy_loss = None
        kl_loss = None



        """
        Update ZF
        """
        if update_critic:
            with torch.no_grad():

                new_next_intents_onehot, new_next_prices_onehot, new_next_intents_logpi, new_next_prices_logpi, _, _= self.get_action_full_information(self.target_policy, batch_next_uttr, batch_next_state)

                new_next_actions = torch.cat((new_next_intents_onehot, new_next_prices_onehot), dim=1).to(torch.float32)
                next_tau, next_tau_hat, next_presum_tau = self.get_tau(batch_size, fp=self.target_fp, device=device)
                target_z1_values = self.target_zf1(batch_next_uttr, batch_next_state, new_next_actions, next_tau_hat)
                target_z2_values = self.target_zf2(batch_next_uttr, batch_next_state, new_next_actions, next_tau_hat)
                if is_pretrain:
                    target_z_values = torch.min(target_z1_values, target_z2_values) # - alpha * new_next_intents_logpi - 0.2 * alpha * new_next_prices_logpi
                else:
                    target_z_values = torch.min(target_z1_values, target_z2_values)  - 0.1 * alpha * new_next_intents_logpi - alpha * new_next_prices_logpi

                z_target = self.reward_scale * batch_reward + (1. - batch_done) * self.discount * target_z_values

            tau, tau_hat, presum_tau = self.get_tau(batch_size, fp=self.fp, device=device)

            z1_pred = self.zf1(batch_uttr, batch_state, actions, tau_hat)
            z2_pred = self.zf2(batch_uttr, batch_state, actions, tau_hat)
            # print("criic", z1_pred[0].data, z2_pred[0].data)
            zf1_loss = self.zf_criterion(z1_pred, z_target, tau_hat, next_presum_tau)
            zf2_loss = self.zf_criterion(z2_pred, z_target, tau_hat, next_presum_tau)

            self.zf1_optimizer.zero_grad()
            zf1_loss.backward()
            self.zf1_optimizer.step()

            self.zf2_optimizer.zero_grad()
            zf2_loss.backward()
            self.zf2_optimizer.step()


        """
        Update Policy
        """
        if update_policy:
            risk_param = self.risk_schedule(self._n_train_steps_total)

            if self.risk_type == 'VaR':
                tau_ = ptu.ones_like(batch_reward) * risk_param
                q1_new_actions = self.zf1(batch_uttr, batch_state, new_actions, tau_)
                q2_new_actions = self.zf2(batch_uttr, batch_state, new_actions, tau_)
            else:
                with torch.no_grad():
                    new_tau, new_tau_hat, new_presum_tau = self.get_tau(batch_size, fp=self.fp, device=device)



                z1_new_actions = self.zf1(batch_uttr, batch_state,new_actions, new_tau_hat)
                z2_new_actions = self.zf2(batch_uttr, batch_state, new_actions, new_tau_hat)
                if self.risk_type in ['neutral', 'std']:
                    q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdim=True)
                    q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdim=True)
                    # print(z1_new_actions[0], z2_new_actions[0])

                    if self.risk_type == 'std':
                        q1_std = new_presum_tau * (z1_new_actions - q1_new_actions).pow(2)
                        q2_std = new_presum_tau * (z2_new_actions - q2_new_actions).pow(2)
                        q1_new_actions -= risk_param * q1_std.sum(dim=1, keepdim=True).sqrt()
                        q2_new_actions -= risk_param * q2_std.sum(dim=1, keepdim=True).sqrt()
                else:
                    with torch.no_grad():
                        risk_weights = distortion_de(new_tau_hat, self.risk_type, risk_param)
                    q1_new_actions = torch.sum(risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdim=True)
                    q2_new_actions = torch.sum(risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdim=True)
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)

            '''
            caculate kl loss
            '''
            kl = (intent_softmax * torch.log2((intent_softmax + 1e-10)/(sl_intent_softmax + 1e-10))).sum(dim=1)

            # print("alpha * new_intents_log_probabilities", alpha * new_intents_log_probabilities[0])
            # print("0.2 * alpha * new_prices_log_probabilities", 0.2 * alpha * new_prices_log_probabilities[0])
            # print("-q_new_actions", -q_new_actions[0])
            # print("kl", 0.01 * kl[0])
            policy_loss = (0.1 * alpha * new_intents_log_probabilities + alpha * new_prices_log_probabilities - q_new_actions + self.args.kl_coefficient * kl).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_grad = ptu.fast_clip_grad_norm(self.policy.parameters(), self.clip_norm)
            self.policy_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf1, self.target_zf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.zf2, self.target_zf2, self.soft_target_tau)

        self._n_train_steps_total += 1
        if update_policy:
            return zf1_loss, zf2_loss, policy_loss, (0.1 * alpha * new_intents_log_probabilities).mean(), (alpha * new_prices_log_probabilities).mean(), \
                   (-q_new_actions).mean(), (self.args.kl_coefficient*kl).mean()
        else:
            return zf1_loss, zf2_loss, None, None, None, None, None