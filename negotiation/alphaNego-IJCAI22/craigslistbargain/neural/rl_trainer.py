import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.autograd import Variable

from cocoa.neural.rl_trainer import Statistics

from core.controller import Controller
from .utterance import UtteranceBuilder

from tensorboardX import SummaryWriter

from neural.sl_trainer import SLTrainer as BaseTrainer, Statistics, SimpleLoss

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
        self.criterion = nn.MSELoss(reduce=False)

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
        return RLStatistics(loss=loss.mean().item(), n_words=data_num)

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p):
        if len(p.shape) == 1:
            p = p.reshape(1, -1)
        assert len(p.shape) == 2
        policy = torch.softmax(p, dim=-1)
        d = torch.distributions.categorical.Categorical(probs=policy)
        # logp = policy.log()
        # loss = torch.sum(policy.mul(-logp), dim=-1)
        return d.entropy()

    def _stats(self, loss, word_num):
        return Statistics(loss.item(), 0, word_num, 0, 0)

class RLTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin',
                 cuda=False, args=None, ent_coef=0.05, val_coef=1):
        self.agents = agents
        self.scenarios = scenarios

        self.training_agent = training_agent
        self.model = agents[training_agent].env.model
        self.critic = None
        self.train_loss = SimpleLoss(inp_with_sfmx=False, use_pact=True)
        self.critic_loss = SimpleCriticLoss()
        self.tom_loss = SimpleLoss(inp_with_sfmx=False, use_pact=False)
        self.entropy_loss = EntropyLoss()
        self.optim = optim
        self.cuda = cuda

        self.ent_coef = ent_coef
        self.val_coef = val_coef
        self.p_reg_coef = 0
        # self.p_reg_coef = 100

        self.best_valid_reward = None
        self.best_valid_loss = None

        self.all_rewards = [[], []]
        self.reward_func = reward_func

        # Summary writer for tensorboard
        self.writer = SummaryWriter(logdir='logs/{}'.format(args.name))

    def update(self, batch_iter, reward, model, discount=0.95):
        model.train()

        nll = []
        # batch_iter gives a dialogue
        stats = Statistics()
        dec_state = None
        for batch in batch_iter:

            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # if enc_state is not None:
            #     print("state: {}".format(batch, enc_state[0].shape))

            policy, price, pvar = self._run_batch(batch)  # (seq_len, batch_size, rnn_size)
            loss, batch_stats = self._compute_loss(batch, policy=policy, price=(price, pvar), loss=self.train_loss)
            stats.update(batch_stats)

            loss = loss.view(-1)
            nll.append(loss)

            # TODO: Don't backprop fully.
            # if dec_state is not None:
            #     dec_state.detach()

        # print('allnll ', nll)

        nll = torch.cat(nll)  # (total_seq_len, batch_size)

        rewards = [Variable(torch.ones(1, 1)*(reward))]
        for i in range(1, nll.size(0)):
            rewards.append(rewards[-1] * discount)
        rewards = rewards[::-1]
        rewards = torch.cat(rewards)
        # print('rl shapes',nll.shape, rewards.shape)

        if self.cuda:
            loss = nll.view(-1).mul(rewards.view(-1).cuda()).mean()
        else:
            loss = nll.view(-1).mul(rewards.view(-1)).mean()

        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 1.)
        self.optim.step()

        return torch.cat([loss.view(-1),
                          nll.mean().view(-1),
                          torch.ones(1, device=loss.device) * stats.mean_loss(0),
                          torch.ones(1, device=loss.device) * stats.mean_loss(1)], ).view(1, -1).cpu().data.numpy()

    def _compute_loss(self, batch, policy=None, price=None, value=None, oracle=None, loss=None):
        if policy is not None:
            batch_size = batch.size
            act_intent = batch.act_intent
            # lsub = policy - policy.max(dim=-1).values
            # print('logits_sub: ', lsub)
            # p = lsub.exp()
            # p = p / p.sum(dim=-1, keepdim=True)
            # print('value', p, target_intent)
            return loss(policy, price, act_intent.reshape(batch_size, -1), batch.act_price, batch.act_price_mask)
        elif value is not None:
            return loss(value, oracle)

    def _run_batch_critic(self, batch):
        value = self.critic(batch.uttr, batch.state)
        return value

    def update_critic(self, batch_iter, reward, critic, discount=0.95):
        critic.train()

        values = []
        # batch_iter gives a dialogue
        dec_state = None
        for batch in batch_iter:
            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            batch.mask_last_price()
            value = self._run_batch_critic(batch)

            values.append(value.view(-1))

        # print('allnll ', nll)
        rewards = [0]*len(values)
        rewards[-1] = reward
        values = torch.cat(values)  # (total_seq_len, batch_size)
        for i in range(len(rewards)-2, -1, -1):
            rewards[i] += (values[i+1].cpu().item())*discount
            rewards[i] = torch.ones(1)*rewards[i]
        rewards[-1] = torch.ones(1)*rewards[-1]

        rewards = torch.cat(rewards)
        if self.cuda:
            rewards = rewards.cuda()

        loss, stats = self._compute_loss(None, value=values, oracle=rewards, loss=self.critic_loss)

        critic.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.)
        self.optim.step()

        return stats

    def _get_scenario(self, scenario_id=None, split='train'):
        scenarios = self.scenarios[split]
        if scenario_id is None:
            # scenario = random.choice(scenarios)
            scenario_id = random.choice(range(len(scenarios)))
        scenario = scenarios[scenario_id % len(scenarios)]
        return scenario, scenario_id

    def _get_controller(self, scenario, split='train', rate=0.5):
        # Randomize
        if random.random() < rate:
            scenario = copy.deepcopy(scenario)
            scenario.kbs = (scenario.kbs[1], scenario.kbs[0])
        sessions = [self.agents[0].new_session(0, scenario.kbs[0]),
                    self.agents[1].new_session(1, scenario.kbs[1])]
        return Controller(scenario, sessions)



        
    def validate(self, args, valid_critic=False):
        split = 'dev'
        self.model.eval()
        total_stats = RLStatistics()
        # print('='*20, 'VALIDATION', '='*20)
        for scenario in self.scenarios[split][:200]:
            controller = self._get_controller(scenario, split=split)
            controller.sessions[self.training_agent].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            stats = RLStatistics(reward=reward, n_words=1)
            total_stats.update(stats)
        # print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        return total_stats

    def save_best_checkpoint(self, checkpoint, opt, valid_stats):

        path = None
        if opt.model_type == 'reinforce':
            if self.best_valid_reward is None or valid_stats.mean_reward() > self.best_valid_reward:
                self.best_valid_reward = valid_stats.mean_reward()
                path = '{root}/{model}_best.pt'.format(
                            root=opt.model_path,
                            model=opt.model_filename)
        elif opt.model_type == 'critic':
            if self.best_valid_loss is None or valid_stats.mean_loss() < self.best_valid_loss:
                self.best_valid_loss = valid_stats.mean_loss()
                path = '{root}/{model}_best.pt'.format(
                            root=opt.model_path,
                            model=opt.model_filename)

        if path is not None:
            print('Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats):
        path=None
        if opt.model_type == 'reinforce':
            path = '{root}/{model}_reward{reward:.2f}_e{episode:d}.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename,
                        reward=stats.mean_reward(),
                        episode=episode)
        elif opt.model_type == 'critic':
            path = '{root}/{model}_loss{reward:.4f}_e{episode:d}.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename,
                        reward=stats.mean_loss(),
                        episode=episode)
        assert path is not None
        return path

    def update_opponent(self, type=None):
        if type is None:
            types = ['policy', 'critic']
        else:
            types = [type]

        print('update opponent model for {}.'.format(types))
        if 'policy' in types:
            tmp_model_dict = self.agents[self.training_agent].env.model.state_dict()
            self.agents[self.training_agent^1].env.model.load_state_dict(tmp_model_dict)
        if 'critic' in types:
            tmp_model_dict = self.agents[self.training_agent].env.critic.state_dict()
            self.agents[self.training_agent^1].env.critic.load_state_dict(tmp_model_dict)

    def learn(self, args):
        if args.model_type == 'reinforce':
            train_policy = True
            train_critic = False
        elif args.model_type == 'critic':
            train_policy = False
            train_critic = True
        elif args.model_type == 'tom':
            train_policy = False
            train_critic = False


        rewards = [None]*2
        s_rewards = [None]*2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        history_train_losses = [[], []]

        for i in range(args.num_dialogues):
            # Rollout
            scenario, _ = self._get_scenario()
            controller = self._get_controller(scenario, split='train')
            # print('set controller for{} {}.'.format(self.training_agent, controller))
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)

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

            for session_id, session in enumerate(controller.sessions):
                # Only train one agent
                if session_id != self.training_agent:
                    continue

                batch_iter = session.iter_batches()
                T = next(batch_iter)

                if train_policy:
                    loss = self.update(batch_iter, reward, self.model, discount=args.discount_factor)
                    history_train_losses[session_id].append(loss)

                if train_critic:
                    stats = self.update_critic(batch_iter, reward, self.critic, discount=args.discount_factor)
                    critic_report_stats.update(stats)
                    critic_stats.update(stats)

            # print('verbose: ', args.verbose)

            if args.verbose:
                if train_policy or args.model_type == 'tom':
                    from core.price_tracker import PriceScaler
                    for session_id, session in enumerate(controller.sessions):
                        bottom, top = PriceScaler.get_price_range(session.kb)
                        print('Agent[{}: {}], bottom ${}, top ${}'.format(session_id, session.kb.role, bottom, top))


                    strs = example.to_text()
                    for str in strs:
                        print(str)
                    print("reward: [0]{}\nreward: [1]{}".format(self.all_rewards[0][-1], self.all_rewards[1][-1]))
                    # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            # Save logs on tensorboard
            if (i + 1) % tensorboard_every == 0:
                for j in range(2):
                    self.writer.add_scalar('agent{}/reward'.format(j), np.mean(self.all_rewards[j][-tensorboard_every:]), i)
                    if len(history_train_losses[j]) >= tensorboard_every:
                        tmp = np.concatenate(history_train_losses[j][-tensorboard_every:], axis=0)
                        tmp = np.mean(tmp, axis=0)
                        self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], i)
                        self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[1], i)
                        self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[2], i)
                        self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[3], i)

            if ((i + 1) % args.report_every) == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt
                if args.histogram:
                    sns.set_style('darkgrid')

                if train_policy:
                    for j in range(2):
                        print('agent={}'.format(j), end=' ')
                        print('step:', i, end=' ')
                        print('reward:', rewards[j], end=' ')
                        print('scaled reward:', s_rewards[j], end=' ')
                        print('mean reward:', np.mean(self.all_rewards[j]))
                        if args.histogram:
                            self.agents[j].env.dialogue_generator.get_policyHistogram()

                if train_critic:
                    critic_report_stats.output(i+1, 0, 0, last_time)
                    critic_report_stats = RLStatistics()

                print('-'*10)
                if args.histogram:
                    plt.show()

                last_time = time.time()

            # Save model
            if (i > 0 and i % 100 == 0) and not args.only_run:
                if train_policy:
                    valid_stats = self.validate(args)
                    self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                    self.update_opponent('policy')

                elif train_critic:
                    # TODO: reverse!
                    self.drop_checkpoint(args, i, critic_stats, model_opt=self.agents[self.training_agent].env.model_args)
                    critic_stats = RLStatistics()
                else:
                    valid_stats = self.validate(args)
                    print('valid result: ', valid_stats.str_loss())


    def _is_valid_dialogue(self, example):
        special_actions = defaultdict(int)
        for event in example.events:
            if event.action in ('offer', 'quit', 'accept', 'reject'):
                special_actions[event.action] += 1
                # Cannot repeat special action
                if special_actions[event.action] > 1:
                    print('Invalid events(0): ')
                    for x in example.events:
                        print('\t', x.action)
                    return False
                # Cannot accept or reject before offer
                if event.action in ('accept', 'reject') and special_actions['offer'] == 0:
                    print('Invalid events(1): ')
                    for x in example.events:
                        print('\t', x.action, x.data)
                    return False
        return True

    def _is_agreed(self, example):
        if example.outcome['reward'] == 0 or example.outcome['offer'] is None:
            return False
        return True

    def _margin_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            # print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # rewards['seller'] = min(rewards['seller'], 1)
        # rewards['seller'] = max(rewards['seller'], -1)
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
       
        return rewards

    def _margin_reward2(self, example):
        # No agreement
        if not self._is_agreed(example):
            # print('No agreement')
            return {'seller': -0.8, 'buyer': -0.8}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        rewards['seller'] = min(rewards['seller'], 1)
        rewards['seller'] = max(rewards['seller'], -1)
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']
        return rewards

    def _length_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        # Encourage long dialogue
        rewards = {}
        for role in ('buyer', 'seller'):
            rewards[role] = len(example.events) / 10.
        return rewards

    def _fair_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        margin_rewards = self._margin_reward(example)
        for role in ('buyer', 'seller'):
            rewards[role] = -1. * abs(margin_rewards[role]) + 2.
        return rewards


    def _balance_reward(self, example):
        # No agreement
        if not self._is_agreed(example):
            # print('No agreement')
            return {'seller': -0.5, 'buyer': -0.5}

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target

        midpoint = (targets['seller'] + targets['buyer']) / 2.

        price = example.outcome['offer']['price']
        norm_factor = abs(midpoint - targets['seller'])
        rewards['seller'] = (price - midpoint) / norm_factor
        # print('calc price: {}\t{}\t{}\t{}'.format(price, midpoint, norm_factor, rewards['seller']))
        # Zero sum
        rewards['buyer'] = -1. * rewards['seller']

        for role in ('buyer', 'seller'):
            rewards[role] += len(example.events) / 20.

        return rewards

    def _base_utility(self, example):

        rewards = {}
        targets = {}
        kbs = example.scenario.kbs
        for agent_id in (0, 1):
            kb = kbs[agent_id]
            targets[kb.role] = kb.target
        if not self._is_agreed(example):
            # print('No agreement')
            rewards = {'seller': -1, 'buyer': -1}
        else:
            price = example.outcome['offer']['price']
            bottom_seller = targets['buyer']
            bottom_buyer = targets['seller']
            rewards['seller'] = (price - bottom_seller) / (targets['seller'] - bottom_seller)
            rewards['buyer'] = (bottom_buyer - price) / (bottom_buyer - targets['buyer'])

            rewards['seller'] = max(rewards['seller'], 0)
            rewards['seller'] = min(rewards['seller'], 1)
            rewards['buyer'] = max(rewards['buyer'], 0)
            rewards['buyer']= min(rewards['buyer'], 1)

        return rewards



    def get_reward(self, example, session):
        if not self._is_valid_dialogue(example):
            print('Invalid')
            rewards = {'seller': -2., 'buyer': -2.}
        else:
            if self.reward_func == 'margin':
                rewards = self._margin_reward(example)
            elif self.reward_func == 'margin2':
                rewards = self._margin_reward2(example)
            elif self.reward_func == 'fair':
                rewards = self._fair_reward(example)
            elif self.reward_func == 'length':
                rewards = self._length_reward(example)
            elif self.reward_func == 'balance':
                rewards = self._balance_reward(example)
            elif self.reward_func == 'base_utility':
                rewards = self._base_utility(example)
        reward = rewards[session.kb.role]
        return reward
