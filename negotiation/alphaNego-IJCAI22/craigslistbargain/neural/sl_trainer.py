from cocoa.neural.trainer import Trainer as BaseTrainer
import onmt
import torch
from torch import nn
from onmt.Utils import use_gpu
from cocoa.io.utils import create_path
from get_policy import PolicyCounter

from tensorboardX import SummaryWriter

import math, time, sys
import numpy as np

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss_intent=0, loss_price=0, n_words=0, n_price=0, n_correct=0):
        self.loss_intent = loss_intent
        self.loss_price = loss_price
        self.n_words = n_words
        self.n_price = n_price
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss_intent += stat.loss_intent
        self.loss_price += stat.loss_price
        self.n_words += stat.n_words
        self.n_price += stat.n_price
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def mean_loss(self, i=2):
        if i == 0:
            return self.loss_intent / self.n_words
        elif i == 1:
            if self.n_price == 0:
                return 0
            return self.loss_price / self.n_price

        return self.mean_loss(0) + self.mean_loss(1)

    def elapsed_time(self):
        return time.time() - self.start_time

    def loss(self):
        return self.loss_intent + self.loss_price

    def ppl(self):
        return math.exp(min(self.loss() / self.n_words, 100))

    def str_loss(self):
        return "acc: %6.4f; loss(act/price/total): %6.4f/%6.4f/%6.4f;" % (self.accuracy(),
               self.mean_loss(0), self.mean_loss(1), self.mean_loss(2))

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (float): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc_intent: %6.4f; loss(act/price/total): %6.4f/%6.4f/%6.4f; " +
               "%6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.mean_loss(0), self.mean_loss(1), self.mean_loss(2),
               time.time() - start))
        sys.stdout.flush()

class Weighted_MSELoss(nn.Module):
    def forward(self, src, x):
        mean = src
        mean = mean.reshape(-1, 1)
        # logstd = logstd.view(-1, 1)
        x = x.reshape(-1, 1)
        # std = logstd.exp()
        # print((1./std)**2)
        # neglogp0 = 0.5 * torch.sum(((x - mean) / std)**2, dim=1)
        # neglogp0 = 0.5 * torch.sum(((x - mean)) ** 2, dim=1)
        # print('p and label: ',x, mean)
        # neglogp1 = 0.5 * np.log(2.0 * np.pi) * x.shape[1]
        # neglogp2 = logstd.sum(dim=1)
        # neglogp = neglogp0 + neglogp1 + neglogp2
        neglogp = 0.5 * torch.sum(((x - mean)) ** 2, dim=1)
        # print('p and label: ', neglogp0)
        # print('neglogp: ', neglogp0, neglogp1, neglogp2)
        return neglogp


class SimpleLoss(nn.Module):
    debug = False

    def __init__(self, inp_with_sfmx=False, use_pact=False):
        super(SimpleLoss, self).__init__()
        if inp_with_sfmx:
            self.criterion_intent = nn.NLLLoss()
        else:
            self.criterion_intent = nn.CrossEntropyLoss(reduction='none')
        self.use_pact = use_pact
        if use_pact:
            self.criterion_price = self.criterion_intent
        else:
            self.criterion_price = Weighted_MSELoss()
        self.use_nll = inp_with_sfmx

    def _get_correct_num(self, enc_policy, tgt_intents):
        # TODO: what is this?
        enc_policy = enc_policy.argmax(dim=1)
        tmp = (enc_policy == tgt_intents).cpu().numpy()
        tgt = tgt_intents.data.cpu().numpy()
        tmp[tgt==19] = 0
        import numpy as np
        return np.sum(tmp), tgt.shape[0]-np.sum(tgt==19)

    def forward(self, enc_policy, enc_price, tgt_policy, tgt_price, pmask=None):
        if pmask is None:
            pmask = torch.ones_like(tgt_price)
        alpha = 1
        tgt_policy = tgt_policy.reshape(-1)
        tgt_price = tgt_price.reshape(-1)
        # print('intent error:', enc_policy, tgt_policy)
        loss0 = self.criterion_intent(enc_policy, tgt_policy)
        # print('loss1: ', enc_price.shape, tgt_price.shape)
        loss1 = self.criterion_price(enc_price, tgt_price).mul(pmask)
        # if SimpleLoss.debug and torch.mean(tgt_price).item() != 1:
        #     print('compaire', torch.cat([enc_price, tgt_price],dim=1))
        #     SimpleLoss.debug = False
        # if pmask.sum() > 0:
        #     loss1 = loss1 * loss1.shape[0] / pmask.sum()
        loss1 = alpha * loss1
        # loss = loss0 + loss1

        correct_num, word_num = self._get_correct_num(enc_policy, tgt_policy)
        price_num = torch.sum(tgt_price).item()
        stats = self._stats(loss0.mean(), loss1.mean(), word_num, price_num, correct_num)
        # if torch.isnan(loss.mean()):
        # print('loss0', loss0)
        # print('loss1', loss1)
        return loss0, loss1, stats

    def _stats(self, loss0, loss1, word_num, price_num, correct_num):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        return Statistics(loss0.item(), loss1.item(), word_num, price_num, correct_num)


class SLTrainer(BaseTrainer):

    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 grad_accum_count=1, args=None):
        # Basic attributes.
        self.model = model
        self.train_loss = \
        self.valid_loss = SimpleLoss(inp_with_sfmx=False)
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.best_valid_loss = None
        self.cuda=False

        # Set model in training mode.
        self.model.train()

        self.use_utterance = False

        # Summary writer for tensorboard
        self.writer = SummaryWriter(logdir='logs/{}'.format(args.name))

    ''' Class that controls the training process which inherits from Cocoa '''

    def _compute_loss(self, batch, policy=None, price=None, loss=None):
        # Get one-hot vectors of target
        # class_num = len(batch.vocab)
        batch_size = len(batch)
        # print('class_num {}\tbatch_size{}'.format(class_num, batch_size))
        # target_intent = torch.zeros(batch_size, class_num)
        # if batch.target_intent.device.type == 'cuda':
        #     target_intent = target_intent.cuda()
        # target_intent = target_intent.scatter_(1, batch.target_intent, 1)
        act_intent = batch.act_intent
        pmean = price
        pmean = pmean.reshape(batch_size, 1).mul(batch.act_price_mask)
        # print('(policy, price, target_intent, batch.target_price)', (policy, price, target_intent, batch.target_price))
        return loss(policy, pmean, act_intent.reshape(batch_size, -1), batch.act_price, batch.act_price_mask)

    def _run_batch(self, batch, dec_state=None, enc_state=None):

        # e_intent, e_price, e_pmask = batch.encoder_intent, batch.encoder_price, batch.encoder_pmask
        # # print('e_intent {}\ne_price{}\ne_pmask{}'.format(e_intent, e_price, e_pmask))
        #
        # if self.use_utterance:
        #     policy, price, pvar = self.model(e_intent, e_price, e_pmask, batch.encoder_dianum, utterance=batch.encoder_tokens)
        # else:
        #     policy, price, pvar = self.model(e_intent, e_price, e_pmask, batch.encoder_dianum, )

        policy, price = self.model(batch.uttr, batch.state)
        return policy, price

    def learn(self, opt, data, report_func):
        """Train model.
        Args:
            opt(namespace)
            model(Model)
            data(DataGenerator)
        """
        print('\nStart training...')
        print(' * number of epochs: %d' % opt.epochs)
        print(' * batch size: %d' % opt.batch_size)

        self.use_utterance = opt.use_utterance

        self.policy_data = data.policy
        self.policy_model = None
        self.incorrect_dist = None

        for epoch in range(opt.epochs):
            print('')
            self.policy_model = PolicyCounter(self.policy_data.counter.shape[0], from_dataset=False)
            self.incorrect_dist = PolicyCounter(self.policy_data.counter.shape[0], from_dataset=False, only_incorrect=True)

            # 1. Train for one epoch on the training set.
            SimpleLoss.debug = True
            train_iter = data.generator('train', cuda=use_gpu(opt))
            train_stats = self.train_epoch(train_iter, opt, epoch, report_func)
            self.writer.add_scalar('train/loss', train_stats.mean_loss(2), epoch)
            self.writer.add_scalar('train/loss_intent', train_stats.mean_loss(0), epoch)
            self.writer.add_scalar('train/loss_price', train_stats.mean_loss(1), epoch)
            self.writer.add_scalar('train/accu', train_stats.accuracy(), epoch)
            print('Train loss: ' + train_stats.str_loss())

            SimpleLoss.debug = True
            # 2. Validate on the validation set.
            valid_iter = data.generator('dev', cuda=use_gpu(opt))
            valid_stats = self.validate(valid_iter)

            self.writer.add_scalar('dev/loss', valid_stats.mean_loss(2), epoch)
            self.writer.add_scalar('dev/loss_intent', valid_stats.mean_loss(0), epoch)
            self.writer.add_scalar('dev/loss_price', valid_stats.mean_loss(1), epoch)
            self.writer.add_scalar('dev/accu', valid_stats.accuracy(), epoch)

            print('Validation loss: ' + valid_stats.str_loss())
            path = '{root}/{model}_policy_e{epoch:d}.png'.format(
                root=opt.model_path,
                model=opt.model_filename,
                epoch=epoch)
            # PolicyCounter.draw_policy([self.policy_data, self.policy_model, self.incorrect_dist], filename=path)

            # 3. Log to remote server.
            # if opt.exp_host:
            #    train_stats.log("train", experiment, optim.lr)
            #    valid_stats.log("valid", experiment, optim.lr)
            # if opt.tensorboard:
            #    train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            #    train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

            # 4. Update the learning rate
            self.epoch_step(valid_stats.ppl(), epoch)

            # 5. Drop a checkpoint if needed.
            if epoch >= opt.start_checkpoint_at:
                self.drop_checkpoint(opt, epoch, valid_stats)

    def train_epoch(self, train_iter, opt, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model back to training mode.
        self.model.train()

        total_stats = Statistics()
        report_stats = Statistics()
        true_batchs = []
        accum = 0
        normalization = 0
        num_batches = next(train_iter)
        self.cuda = use_gpu(opt)

        for batch_idx, batch in enumerate(train_iter):
            true_batchs.append(batch)
            accum += 1

            if accum == self.grad_accum_count:
                self._gradient_accumulation(true_batchs, total_stats, report_stats)
                true_batchs = []
                accum = 0

            if report_func is not None:
                report_stats = report_func(opt, epoch, batch_idx, num_batches,
                                           total_stats.start_time, report_stats)

        # Accumulate gradients one last time if there are any leftover batches
        # Should not run for us since we plan to accumulate gradients at every
        # batch, so true_batches should always equal candidate batches
        if len(true_batchs) > 0:
            self._gradient_accumulation(true_batchs, total_stats, report_stats)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.

        self.model.eval()

        stats = Statistics()

        num_val_batches = next(valid_iter)
        for batch in valid_iter:
            if batch is None:
                continue
            policy, price = self._run_batch(batch, )
            loss0, loss1, batch_stats = self._compute_loss(batch, policy, price, self.train_loss)
            stats.update(batch_stats)

        # Set model back to training mode
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, valid_stats, model_opt=None, score_type='loss', best_only=False):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        if not isinstance(valid_stats, float):
            valid_stats = valid_stats.mean_loss()
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        checkpoint = {
            'model': model_state_dict,
            'opt': opt if not model_opt else model_opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        # Add critic in checkpoint
        if hasattr(self, 'critic'):
            checkpoint['critic'] = self.critic.state_dict()

        if hasattr(self, 'tom'):
            checkpoint['tom'] = self.tom.state_dict()

        path = self.checkpoint_path(epoch, opt, valid_stats, score_type)
        create_path(path)
        if not best_only:
            print('Save checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

        self.save_best_checkpoint(checkpoint, opt, valid_stats, score_type)

        return path

    def save_best_checkpoint(self, checkpoint, opt, valid_stats, score_type='loss'):
        update_best = False
        if score_type == 'loss' and (self.best_valid_loss is None or valid_stats < self.best_valid_loss):
            update_best = True
        if score_type == 'reward' and (self.best_valid_loss is None or valid_stats > self.best_valid_loss):
            update_best = True
        if update_best:
            self.best_valid_loss = valid_stats
            path = '{root}/{model}_best.pt'.format(
                root=opt.model_path,
                model=opt.model_filename)

            print('Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, epoch, opt, stats, score_type='loss'):
        path = '{root}/{model}_{score_type}{score:.4f}_e{epoch:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    score_type=score_type,
                    score=stats,
                    epoch=epoch)
        assert path is not None
        return path

    def _gradient_accumulation(self, true_batchs, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        dec_state = None
        for batch in true_batchs:
            if batch is None:
                continue

            self.model.zero_grad()
            policy, price = self._run_batch(batch)
            self.policy_model.update_from_batch(batch, policy)
            self.incorrect_dist.update_from_batch(batch, policy)
            # print('output is: ', policy, price)

            loss0, loss1, batch_stats = self._compute_loss(batch, policy, price, self.train_loss)
            loss = loss0 + loss1

            # tmp = torch.cat([batch.target_price, price.view(-1,1), batch.target_pmask], dim=1)
            # print('target_price ',tmp)
            # print('loss is: ', loss)
            # print('loss.mean()', loss.mean())
            loss.mean().backward()
            self.optim.step()

            total_stats.update(batch_stats)
            if not isinstance(report_stats, Statistics):
                print('report_', type(report_stats))
            report_stats.update(batch_stats)
