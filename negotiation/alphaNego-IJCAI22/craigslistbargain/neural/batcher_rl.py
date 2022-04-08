import numpy as np
from itertools import zip_longest

import torch
from torch.autograd import Variable

from .symbols import markers

def pad_list_to_array(l, fillvalue, dtype):
    '''
    l: list of lists with unequal length
    return: np array with minimal padding
    '''
    return np.array(list(zip_longest(*l, fillvalue=fillvalue)), dtype=dtype).T

class Batch(object):

    attrs_name = ['encoder_intent', 'encoder_price', 'encoder_tokens', 'encoder_pmask',
                  'target_intent', 'target_price', 'target_pmask',
                  'title_inputs', 'desc_inputs', 'encoder_dianum']

    @staticmethod
    def merge_batches(batches):
        assert isinstance(list, batches)
        attrs_value = []
        for i, name in enumerate(Batch.attrs_name):
            attrs_value.append([])
            for b in batches:
                attrs_value[i].append(getattr(b, name))

        for i, value in enumerate(attrs_value):
            need_pad = False
            max_length = value[0].shape[1]
            for v in value:
                if v.shape[1] != max_length:
                    need_pad = True
                max_length = max(max_length, v.shape[1])
            if need_pad:
                for j, v in enumerate(value):
                    if v.shape[1] == max_length:
                        continue
                    value[j] = torch.nn.functional.pad(v, (0, max_length-v.shape[1]), 'constant', 0)

            attrs_value[i] = torch.cat(value, dim=0)

        return Batch(Batch.attrs_name, attrs_value)

    def convert_device(self, device):
        for v in self.tensor_attributes:
            tmp = getattr(self, v)
            if device != tmp.device:
                setattr(self, v, tmp.to(device))
        # if device == self.device:
        #     return
        # self.device = device
        #
        # self.encoder_tokens = self.encoder_tokens.to(device)
        # self.encoder_intent = self.encoder_intent.to(device)
        # self.encoder_price = self.encoder_price.to(device)
        # self.encoder_pmask = self.encoder_pmask.to(device)
        #
        # self.target_intent = self.target_intent.to(device)
        # self.target_price = self.target_price.to(device)
        # self.target_pmask = self.target_pmask.to(device)
        # self.policy_mask = self.policy_mask.to(device)
        #
        # self.title_inputs = self.title_inputs.to(device)
        # self.desc_inputs = self.desc_inputs.to(device)
        # if self.encoder_dianum is not None:
        #     self.encoder_dianum = self.encoder_dianum.to(device)



    @staticmethod
    def int_to_onehot(tensor, onehot_size):
        batch_size = tensor.shape[0]
        tensor = tensor.reshape(-1, 1)
        real_size = tensor.shape[0]
        onehot = torch.zeros((real_size, onehot_size), device=tensor.device).scatter(1, tensor, 1)
        return onehot.reshape(batch_size, -1)
    
    def reshape_tensors(self, attributes):
        # Reshape all the tensors except uttr
        for v in attributes:
            tmp = getattr(self, v)
            d = len(tmp.shape)
            if d == 1:
                tmp = tmp.reshape(1, -1)
            if d > 2:
                tmp = tmp.reshape(tmp.shape[0], -1)
            setattr(self, v, tmp)

    @staticmethod
    def get_policy_mask(intents, vocab):
        policy_mask = np.ones((len(intents), len(vocab)))
        offer_idx = vocab.to_ind('offer')
        acc_idx = vocab.to_ind('accept')
        rej_idx = vocab.to_ind('reject')
        none_idx = vocab.to_ind('None')
        unk_idx = vocab.to_ind('unknown')
        quit_idx = vocab.to_ind('quit')
        pad_idx = vocab.to_ind(markers.PAD)
        start_idx = vocab.to_ind('start')

        for i in range(len(intents)):
            if intents[i] == offer_idx:
                policy_mask[i, :] = 0
                policy_mask[i, [acc_idx, rej_idx]] = 1
            elif intents[i] in [acc_idx, rej_idx]:
                policy_mask[i, :] = 0
                policy_mask[i, [quit_idx, ]] = 1
            else:
                policy_mask[i, [acc_idx, rej_idx, pad_idx, none_idx, unk_idx, quit_idx, start_idx]] = 0

        return policy_mask

    @classmethod
    def convert_data(cls, encoder_args, decoder_args, context_data, vocab, cuda=False):
        encoder_intent = encoder_args['intent']
        encoder_price = encoder_args['price']
        encoder_pmask = encoder_args['price_mask']
        encoder_tokens = context_data['encoder_tokens']

        encoder_extra = encoder_args['extra']

        only_run = False
        if decoder_args is None:
            only_run = True


        # if not for_value:
        if not only_run:
            target_intent = decoder_args['intent']
            target_price = decoder_args['price']
            target_pmask = decoder_args['price_mask']
            target_pact = decoder_args['price_act']
            target_prob = decoder_args['prob']
        else:
            target_intent, target_price, target_pmask, target_pact, target_prob = None, None, None, None, None
        # else:
        #     self.target_value = decoder_args['value']

        # title_inputs = decoder_args['context']['title']
        # desc_inputs = decoder_args['context']['description']

        size = len(encoder_intent)
        context_data = context_data

        batch_major_attributes = ['encoder_intent', 'encoder_price', 'encoder_pmask',
                                  'title_inputs', 'desc_inputs']

        batch_major_attributes += ['encoder_dianum']

        # if not for_value:
        batch_major_attributes += ['target_intent', 'target_price']

        # To tensor/variable
        # self.encoder_tokens = self.pad_tokens(self.encoder_tokens)
        tokens = []
        for i, t in enumerate(encoder_tokens):
            tokens.append(cls.to_variable(t, 'long', cuda).reshape(1, -1))
        encoder_tokens = tokens

        encoder_intent = cls.to_variable(encoder_intent, 'long', cuda)
        encoder_intent = cls.int_to_onehot(encoder_intent, len(vocab))
        encoder_price = cls.to_variable(encoder_price, 'float', cuda)
        encoder_pmask = cls.to_variable(encoder_pmask, 'float', cuda)

        encoder_extra = cls.to_variable(encoder_extra, 'float', cuda).reshape(-1, 5)

        # if not only_run:
        target_intent = cls.to_variable(target_intent, 'long', cuda)
        # target_intent = cls.int_to_onehot(target_intent, len(vocab))
        target_price = cls.to_variable(target_price, 'float', cuda)
        target_pmask = cls.to_variable(target_pmask, 'float', cuda)
        target_pact = cls.to_variable(target_pact, 'long', cuda)
        target_prob = cls.to_variable(target_prob, 'float', cuda)
        # policy_mask = cls.to_variable(policy_mask, 'float', cuda)
        # print('ti1, ', self.target_intent)

        # Reshape all the tensors except uttr
        # for v in batch_major_attributesa:
        #     tmp = getattr(self, v)
        #     d = len(tmp.shape)
        #     if d == 1:
        #         tmp = tmp.reshape(1, -1)
        #     if d > 2:
        #         tmp = tmp.reshape(tmp.shape[0], -1)
        #     setattr(self, v, tmp)

        # else:
        #     self.target_value = cls.to_variable(self.target_value, 'float', cuda).unsqueeze(1)

        # self.title_inputs = cls.to_variable(self.title_inputs, 'long', cuda)
        # self.desc_inputs = cls.to_variable(self.desc_inputs, 'long', cuda)

        # for i in batch_major_attributes:
        #     print('{} size: {}'.format(i, getattr(self, i).shape))
        # exit()

        state_sentence = torch.cat([encoder_intent, encoder_price], dim=-1)
        state_extra = encoder_extra
        state_obs = torch.cat([encoder_intent, encoder_price.mul(encoder_pmask), encoder_pmask], dim=-1)
        uttr = encoder_tokens

        return (state_sentence, state_extra, state_obs), uttr, \
               (target_intent, target_price, target_pact, target_pmask), target_prob

    # def __init__(self, attrs_name, attrs_value, for_value=False):
    #     for i, name in enumerate(attrs_name):
    #         setattr(self, name, attrs_value[i])
    #     self.for_value = for_value
    #     self.device = self.encoder_intent.device

    def __init__(self, encoder_args, decoder_args, context_data, vocab,
                time_major=True, num_context=None, cuda=False, for_value=False, msgs=None):
        self.tensor_attributes = []
        self.msgs = msgs
        vocab = vocab
        self.num_context = num_context
        self.encoder_intent = encoder_args['intent']
        self.encoder_price = encoder_args['price']
        self.encoder_pmask = encoder_args['price_mask']
        self.encoder_tokens = context_data['encoder_tokens']

        use_dianum = False
        if encoder_args.get('dia_num') is not None:
            use_dianum = True

        if use_dianum:
            self.encoder_dianum = encoder_args['dia_num']
        else:
            self.encoder_dianum = None

        self.for_value = for_value

        # if not for_value:
        self.target_intent = decoder_args['intent']
        self.target_price = decoder_args['price']
        self.target_pmask = decoder_args['price_mask']
        # TODO: Get policy mask from the intent
        self.policy_mask = np.ones((len(self.encoder_intent), len(vocab)))
        offer_idx = vocab.to_ind('offer')
        acc_idx = vocab.to_ind('accept')
        rej_idx = vocab.to_ind('reject')
        none_idx = vocab.to_ind('None')
        unk_idx = vocab.to_ind('unknown')
        quit_idx = vocab.to_ind('quit')
        pad_idx = vocab.to_ind(markers.PAD)
        start_idx = vocab.to_ind('start')

        for i in range(len(self.encoder_intent)):
            if self.encoder_intent[i][-1] == offer_idx:
                self.policy_mask[i, :] = 0
                self.policy_mask[i,[acc_idx, rej_idx]] = 1
            elif self.encoder_intent[i][-1] in [acc_idx, rej_idx]:
                self.policy_mask[i, :] = 0
                self.policy_mask[i, [quit_idx, ]] = 1
            else:
                self.policy_mask[i, [acc_idx, rej_idx, pad_idx, none_idx, unk_idx, quit_idx, start_idx]] = 0
        # else:
        #     self.target_value = decoder_args['value']

        self.title_inputs = decoder_args['context']['title']
        self.desc_inputs = decoder_args['context']['description']

        self.size = len(self.encoder_intent)
        self.context_data = context_data

        unsorted_attributes = ['encoder_intent', 'encoder_price', 'encoder_tokens', 'encoder_pmask',
                               'title_inputs', 'desc_inputs']
        batch_major_attributes = ['encoder_intent', 'encoder_price', 'encoder_tokens', 'encoder_pmask',
                                  'title_inputs', 'desc_inputs']

        if use_dianum:
            unsorted_attributes += ['encoder_dianum']
            batch_major_attributes += ['encoder_dianum']

        # if not for_value:
        unsorted_attributes += ['target_intent', 'target_price']
        batch_major_attributes += ['target_intent', 'target_price']

        # else:
        #     unsorted_attributes += ['target_value']
        #     batch_major_attributes += ['target_value']

        # if num_context > 0:
        #     self.context_inputs = encoder_args['context'][0]
        #     unsorted_attributes.append('context_inputs')
        #     batch_major_attributes.append('context_inputs')

        # self.lengths, sorted_ids = self.sort_by_length(self.encoder_inputs)
        # self.tgt_lengths, _ = self.sort_by_length(self.decoder_inputs)

        # if time_major:
        #     for attr in batch_major_attributes:
        #         print(attr)
        #         if len(getattr(self, attr)) > 0:
        #             setattr(self, attr, np.swapaxes(getattr(self, attr), 0, 1))

        # To tensor/variable
        self.encoder_tokens = self.pad_tokens(self.encoder_tokens)
        self.encoder_tokens = self.to_variable(self.encoder_tokens, 'long', cuda)
        self.encoder_intent = self.to_variable(self.encoder_intent, 'long', cuda)
        self.encoder_price = self.to_variable(self.encoder_price, 'float', cuda)
        self.encoder_pmask = self.to_variable(self.encoder_pmask, 'float', cuda)

        if use_dianum:
            self.encoder_dianum = self.to_variable(self.encoder_dianum, 'float', cuda).view(-1,3)
        else:
            self.encoder_dianum = None

        # if not for_value:
        #     # print('ti0, ', self.target_intent)
        self.target_intent = self.to_variable(self.target_intent, 'long', cuda)
        self.target_price = self.to_variable(self.target_price, 'float', cuda)
        self.target_pmask = self.to_variable(self.target_pmask, 'float', cuda)
        self.policy_mask = self.to_variable(self.policy_mask, 'float', cuda)
        # print('ti1, ', self.target_intent)
        for v in ['target_intent', 'target_price', 'target_pmask']:
            while True:
                d = len(getattr(self, v).shape)
                if d >= 2:
                    break
                setattr(self, v, getattr(self, v).unsqueeze(d))
        # print('ti2, ', self.target_intent)

        # else:
        #     self.target_value = self.to_variable(self.target_value, 'float', cuda).unsqueeze(1)

        self.title_inputs = self.to_variable(self.title_inputs, 'long', cuda)
        self.desc_inputs = self.to_variable(self.desc_inputs, 'long', cuda)
        # self.targets = self.to_variable(self.targets, 'long', cuda)
        # self.lengths = self.to_tensor(self.lengths, 'long', cuda)
        # self.tgt_lengths = self.to_tensor(self.tgt_lengths, 'long', cuda)
        # if num_context > 0:
        #     self.context_inputs = self.to_variable(self.context_inputs, 'long', cuda)

        for i in batch_major_attributes:
            print('{} size: {}'.format(i, getattr(self, i).shape))
        exit()

        self.device = self.encoder_intent.device()
        vocab = None

    @staticmethod
    def pad_tokens(encoder_tokens):
        max_len = 0
        new_tokens = []
        for t in encoder_tokens:
            max_len = max(max_len, len(t))
        for t in encoder_tokens:
            new_tokens.append(t.copy())
            while len(new_tokens[-1]) < max_len:
                new_tokens[-1].append(0)
        return new_tokens

    @classmethod
    def to_tensor(cls, data, dtype, cuda=False):
        if type(data) == np.ndarray:
            data = data.tolist()
        if data is None or (isinstance(data, list) and data[0] is None):
            return None
        if dtype == "long":
            tensor = torch.tensor(data, dtype=torch.int64)
        elif dtype == "float":
            tensor = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError
        tensor = cls.reshape_tensor(tensor)
        return tensor.cuda() if cuda else tensor

    @staticmethod
    def reshape_tensor(tensor):
        d = len(tensor.shape)
        if d == 1:
            tensor = tensor.reshape(1, -1)
        if d > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        return tensor

    @classmethod
    def to_variable(cls, data, dtype, cuda=False):
        if data is None:
            return data
        tensor = cls.to_tensor(data, dtype)
        var = Variable(tensor)
        return var.cuda() if cuda else var

    def sort_by_length(self, inputs):
        """
        Args:
            inputs (numpy.ndarray): (batch_size, seq_length)
        """
        pad = 0
        def get_length(seq):
            for i, x in enumerate(seq):
                if x == pad:
                    return i
            return len(seq)
        lengths = [get_length(s) for s in inputs]
        # TODO: look into how it works for all-PAD seqs
        lengths = [l if l > 0 else 1 for l in lengths]
        sorted_id = np.argsort(lengths)[::-1]
        return lengths, sorted_id

    def order_by_id(self, inputs, ids):
        if ids is None:
            return inputs
        else:
            if type(inputs) is np.ndarray:
                if len(inputs) == 0: return inputs
                return inputs[ids, :]
            elif type(inputs) is list:
                return [inputs[i] for i in ids]
            else:
                raise ValueError('Unknown input type {}'.format(type(inputs)))

    def mask_last_price(self):
        self.encoder_price[:,-1] = 0


class RawBatch(Batch):

    def to(self, device):
        if device is None:
            return
        for k in self.__dict__:
            v = getattr(self, k)
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device=device))
            elif isinstance(v, tuple) and (k == 'state' or k == 'act'):
                setattr(self, k, tuple([vv.to(device) for vv in v]))
            elif isinstance(v, list) and k == 'uttr':
                setattr(self, k, [vv.to(device) for vv in v])


    @staticmethod
    def merge(batches):
        names = ['state', 'uttr', 'act', 'prob']
        val = []
        for k in names:
            tmp = []
            for b in batches:
                v = getattr(b, k)
                if v is None:
                    tmp = None
                elif isinstance(v, tuple):
                    if len(tmp) < len(v):
                        tmp = [[] for _ in range(len(v))]
                    for i in range(len(v)):
                        tmp[i].append(v[i])
                elif isinstance(v, list):
                    tmp = tmp + v
                elif isinstance(v, torch.Tensor):
                    tmp.append(v)
                else:
                    print('unexpected v:', k, v)

            if tmp is None:
                tmp = None
            elif isinstance(tmp, list):
                if k != 'uttr':
                    # state[2]
                    if isinstance(tmp[0], list):
                        for i in range(len(tmp)):
                            tmp[i] = torch.cat(tmp[i], dim=0)
                        tmp = tuple(tmp)
                    elif isinstance(tmp[0], torch.Tensor):
                        tmp = torch.cat(tmp, dim=0)

            val.append(tmp)

        return RawBatch(**{k:val[i] for i, k in enumerate(names)})

    def get_pre_info(self, lf_vocab):
        intent_size = lf_vocab.size
        sentence, extra = self.state_0, self.state_1
        state_length = self.state_0.shape[1] // (intent_size+1)
        intents = sentence[:, (state_length-1)*intent_size: state_length*intent_size]
        prices = extra[:, -2:]
        intents = list(intents.argmax(dim=1).reshape(-1).cpu().numpy())
        return intents, prices

    def policy_mask(self, vocab):
        intents, _ = self.get_pre_info(vocab)
        return self.get_policy_mask(intents, vocab)

    @staticmethod
    def init_vocab(vocab):
        if hasattr(RawBatch, 'intent_size'):
            return
        RawBatch.offer_idx = vocab.to_ind('offer')
        RawBatch.acc_idx = vocab.to_ind('accept')
        RawBatch.rej_idx = vocab.to_ind('reject')
        RawBatch.none_idx = vocab.to_ind('None')
        RawBatch.unk_idx = vocab.to_ind('unknown')
        RawBatch.quit_idx = vocab.to_ind('quit')
        RawBatch.pad_idx = vocab.to_ind(markers.PAD)
        RawBatch.start_idx = vocab.to_ind('start')
        RawBatch.intent_size = len(vocab)
        RawBatch.extra_size = 5 # Role | num | last_prices

        # print('RB idxes:', RawBatch.offer_idx, RawBatch.acc_idx, RawBatch.rej_idx)
        # print('RB idxes:', RawBatch.quit_idx, RawBatch.start_idx, RawBatch.none_idx)
        # exit()

    @staticmethod
    def get_policy_mask(intents, vocab):
        # if isinstance(intents, torch.Tensor):
        RawBatch.init_vocab(vocab)
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
        # print('policy_mask: ', policy_mask)
        # exit()
        return policy_mask

    @classmethod
    def generate(cls, encoder_args, decoder_args, context_data, vocab, cuda=False):
        state, uttr, act, prob \
            = cls.convert_data(encoder_args=encoder_args, decoder_args=decoder_args, context_data=context_data,
                     vocab=vocab, cuda=cuda)
        build_dict = {'state': state, 'uttr': uttr, 'act': act, 'prob': prob}
        return cls(**build_dict)

    @property
    def size(self):
        if getattr(self, 'state', None) is None and getattr(self, 'state_0', None) is None:
            return 0
        if (not hasattr(self, 'state')) or isinstance(self.state, tuple):
            return self.state_0.shape[0]
        else:
            return self.state.shape[0]

    def __init__(self, **kwargs):
        """ Special rules: tuple elements will be storage separately.
        uttr data format: [Tensor]
        other: Tensor or list
            TODO: use Tensor rather than list

        """
        self.tensor_attributes = []
        for k in kwargs:
            if isinstance(kwargs[k], tuple):
                # TODO: remove redundant code
                # redundant but necessary for now
                self.__setattr__(k, kwargs[k])

                for i, v in enumerate(kwargs[k]):
                    new_k = "{}_{}".format(k, i)
                    self.tensor_attributes.append(new_k)
                    self.__setattr__(new_k, v)
            else:
                self.tensor_attributes.append(k)
                self.__setattr__(k, kwargs[k])


class RLBatch(RawBatch):

    def get_pre_info(self, lf_vocab):
        intent_size = lf_vocab.size
        sentence, extra = self.state[:, :-5], self.state[:, -5:]
        state_length = sentence.shape[1] // (intent_size+1)
        intents = sentence[:, (state_length-1)*intent_size: state_length*intent_size]
        prices = extra[:, -2:]
        intents = list(intents.argmax(dim=1).reshape(-1).cpu().numpy())
        return intents, prices

    @classmethod
    def from_raw(cls, rawBatch, reward, done):
        uttr, prob =rawBatch.uttr, rawBatch.prob
        state = (rawBatch.state_0, rawBatch.state_1, rawBatch.state_2)
        act = (rawBatch.act_0, rawBatch.act_1, rawBatch.act_2, rawBatch.act_3)
        # super(RLBatch, self).__init__()
        state = torch.cat(state[:2], dim=-1)
        prob = prob
        uttr = uttr
        act_intent = act[0]
        act_price = act[2]
        act_price_mask = act[3]
        # self.value = []
        reward = reward
        done = done
        build_dict = {'state': state, 'uttr': uttr, 'act_intent': act_intent,
                      'act_price': act_price, 'act_price_mask':act_price_mask,
                      'reward': reward, 'done': done, 'prob':prob}
        return cls(**build_dict)
        # self.tensor_attributes = ['state', 'uttr', 'act_intent', 'act_price', 'act_price_mask']

    def __len__(self):
        return self.state.shape[0]


class ToMBatch(RawBatch):
    @classmethod
    def from_raw(cls, rawBatch, strategy):
        uttr, prob =rawBatch.uttr, rawBatch.prob
        state = (rawBatch.state_0, rawBatch.state_1, rawBatch.state_2)
        act = (rawBatch.act_0, rawBatch.act_1, rawBatch.act_2, rawBatch.act_3)
        # print('i_s:', -(RawBatch.intent_size+1)*2)
        # state = (state[0][:, -(RawBatch.intent_size+1)*2:], state[1])
        length = state[2].shape[1] // (RawBatch.intent_size+2)
        intent = state[2][:, :length*RawBatch.intent_size][:, -RawBatch.intent_size*2:]
        price = state[2][:, length*RawBatch.intent_size:length*(RawBatch.intent_size+1)][:, -2:]
        pmask = state[2][:, length*(RawBatch.intent_size+1):length*(RawBatch.intent_size+2)][:, -2:]
        # One step state
        identity_state = torch.cat([intent, price, pmask], dim=-1)
        # self.identity_state = state[2][:, -(RawBatch.intent_size+2)*2:]
        extra = state[1][:, :3]
        # self.state = torch.cat([self.identity_state, self.extra], dim=-1)
        # # Ignore last prices, but use multi-steps state
        # self.original_state = torch.cat([self.extra, state[2]], dim=-1)
        last_price = state[1][:, -2:]
        state = state[2]
        uttr = uttr
        act_intent = act[0]
        act_price = act[1]
        act_price_mask = act[3]

        strategy = cls.to_tensor(strategy, 'long', cuda=state.device.type!='cpu')
        strategy = Batch.int_to_onehot(strategy.reshape(-1), 7)

        tensor_attributes = ['state', 'identity_state', 'extra', 'uttr', 'last_price',
                                  'act_intent', 'act_price', 'act_price_mask', 'strategy']
        build_dict = {
            'state': state,
            'identity_state': identity_state,
            'extra': extra,
            'uttr': uttr,
            'act_intent': act_intent,
            'act_price': act_price,
            'act_price_mask': act_price_mask,
            'last_price': last_price,
            'strategy': strategy,
        }
        return cls(**build_dict)

    def get_pre_info(self, lf_vocab):
        intent_size = lf_vocab.size
        sentence = self.state
        state_length = sentence.shape[1] // (intent_size+1)
        intents = sentence[:, (state_length-1)*intent_size: state_length*intent_size]
        prices = self.last_price
        intents = list(intents.argmax(dim=1).reshape(-1).cpu().numpy())
        return intents, prices

    def __len__(self):
        return self.state.shape[0]


class SLBatch(Batch):

    def __len__(self):
        return self.state.shape[0]

    def __init__(self, encoder_args, decoder_args, context_data, vocab, cuda=False):
        # super(SLBatch, self).__init__()
        state, uttr, act, prob \
            = self.convert_data(encoder_args=encoder_args, decoder_args=decoder_args, context_data=context_data,
                     vocab=vocab, cuda=cuda)
        self.state = torch.cat(state[:2], dim=-1)
        self.uttr = uttr
        self.act_intent = act[0]
        self.act_price = act[1]
        self.act_price_mask = act[3]
        # self.size = state[0].shape[0]

        self.tensor_attributes = ['state', 'uttr', 'act_intent', 'act_price', 'act_price_mask']


class DialogueBatcher(object):
    def __init__(self, kb_pad=None, mappings=None, model='seq2seq', num_context=2, state_length=2, dia_num=None):
        self.pad = mappings['utterance_vocab'].to_ind(markers.PAD)
        self.kb_pad = kb_pad
        self.mappings = mappings
        self.model = model
        self.num_context = num_context
        self.state_length = state_length
        if dia_num == 0:
            dia_num = None
        self.dia_num = dia_num

    def _normalize_dialogue(self, dialogues):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([d.num_lfs for d in dialogues])
        for dialogue in dialogues:
            dialogue.pad_turns(max_num_turns)
        num_turns = dialogues[0].num_turns
        return num_turns

    # i == -1: the last item
    # attached_events: for tom search process
    def _get_turn_batch_at(self, dialogues, STAGE, i, step_back=1, attached_events=None):
        # For RL setting, we use lf as turns.
        if STAGE == 0:
            # encoder pad
            pad = self.mappings['src_vocab'].to_ind(markers.PAD)
        else:
            # decoder & target pad
            pad = self.mappings['tgt_vocab'].to_ind(markers.PAD)

        attached_role, attached_lfs, attached_uttr = None, None, None
        attached_num = 0
        if attached_events is not None:
            attached_lfs, attached_role, attached_uttr = attached_events
            attached_num = len(attached_lfs)

        if i == -1:
            i = len(dialogues[0].lfs)-1
            if attached_events is not None:
                i = i + 1

        if i is None:
            # Return all turns
            # Format: {'intent': [0, 1, ...], 'price': [0.0, 0.5, ...]}
            tmp = [self._get_turn_batch_at(dialogues, STAGE, i, step_back=step_back) for i in range(dialogues[0].num_turns)]
            turns = {'intent': [], 'price': []}
            for k in turns:
                for j in tmp:
                    turns[k].append(j[k])
            return turns
        else:
            if STAGE == 1:
                # Tokens
                tokens = []
                for d in dialogues:
                    # print('echo d.tokens: ', i, len(d.tokens), d.tokens)
                    if attached_num > 0:
                        for uttr in attached_uttr:
                            tokens.append(uttr)
                    else:
                        if i >= len(d.tokens):
                            break
                            # print('all length', i, len(d.tokens), len(d.token_turns), len(d.lf_tokens), len(d.lfs))
                        tokens.append(d.tokens[i])
                return tokens
            elif STAGE == 2:
                pact = []
                prob = []
                for d in dialogues:
                    if i >= len(d.price_actions):
                        break
                    pact.append(d.price_actions[i].get('price_act'))
                    prob.append(d.price_actions[i].get('prob'))
                return {'price_act':pact, 'prob':prob}

            elif STAGE == 0:
                # STAGE == ENC: encoder batch
                intents = []
                prices = []
                oprices = []

                def get_turns():
                    intent = [self.pad]*(max(0, step_back-i-1))
                    price = [None]*(max(0, step_back-i-1))
                    oprice = [None] * (max(0, step_back - i - 1))
                    for j in range(max(0, i + 1 - step_back), i + 1):
                        tmp = lfs[j]
                        intent.append(tmp['intent'])
                        price.append(tmp.get('price'))
                        # price.append(tmp.get('price'))
                        oprice.append(tmp.get('original_price'))
                    # tmp = d.lfs[i]
                    # intent = np.array(intent)
                    # price = np.array(price)
                    intents.append(intent)
                    prices.append(price)
                    oprices.append(oprice)

                for d in dialogues:
                    if attached_lfs is not None:
                        lfs = d.lfs.copy()
                        lfs.append(None)
                        for e in attached_lfs:
                            # print('lfs: ', lfs)
                            lfs[-1] = e
                            # print('lfs: ', lfs)
                            get_turns()

                    else:
                        lfs = d.lfs
                        if len(lfs) == 0:
                            print('tt', len(d.lfs), len(d.lf_turns), len(d.token_turns), len(d.tokens))
                        get_turns()

                # if step_back == 2:
                #     print(step_back-i-1, i+1-step_back, i+1, intents)
                turns = {'intent': intents, 'price': prices, 'original_price': oprices}
                return turns
            else:
                # STAGE==ROLE: role batch
                roles = []

                def get_roles():
                    if droles[i] is None:
                        roles.append([0, 0])
                    else:
                        tmp = [0, 0]
                        tmp[droles[i]] = 1
                        roles.append(tmp)
                for d in dialogues:
                    if attached_events is not None:
                        droles = d.rid.copy()
                        droles.append(attached_role)
                        for j in range(attached_num):
                            get_roles()
                    else:
                        droles = d.rid
                        get_roles()
                return roles

    def create_context_batch(self, dialogues, pad):
        category_batch = np.array([d.category for d in dialogues], dtype=np.int32)
        # TODO: make sure pad is consistent
        #pad = Dialogue.mappings['kb_vocab'].to_ind(markers.PAD)
        title_batch = pad_list_to_array([d.title for d in dialogues], pad, np.int32)
        # TODO: hacky: handle empty description
        description_batch = pad_list_to_array([[pad] if not d.description else d.description for d in dialogues], pad, np.int32)
        return {
                'category': category_batch,
                'title': title_batch,
                'description': description_batch,
                }

    def _get_agent_batch_at(self, dialogues, i):
        for d in dialogues:
            if i >= len(d.agents):
                print(d.agents, i)
                print(d.lf_turns)
                assert False
        return [dialogue.agents[i] for dialogue in dialogues]

    def _get_kb_batch(self, dialogues):
        return [dialogue.kb for dialogue in dialogues]

    def _remove_last(self, array, value, pad):
        array = np.copy(array)
        nrows, ncols = array.shape
        for i in range(nrows):
            for j in range(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = pad
                    break
        return array

    def _remove_prompt(self, input_arr):
        '''
        Remove starter symbols (e.g. <go>) used for the decoder.
        input_arr: (batch_size, seq_len)
        '''
        # TODO: depending on prompt length
        return input_arr[:, 1:]

    def get_encoder_inputs(self, encoder_turns):
        intent = encoder_turns['intent']
        size = len(encoder_turns['price'])
        price = []
        price_mask = []
        # print(intent)
        for i in range(size):
            price.append([p if p is not None else (j+1) % 2 for j, p in enumerate(encoder_turns['price'][i])])
            price_mask.append([1 if p is not None else 0 for p in encoder_turns['original_price'][i]])
        # intent = np.array(intent, dtype=np.int32)
        # price = np.array(price, dtype=np.float)
        # price_mask = np.array(price_mask, dtype=np.int32)
        return intent, price, price_mask

    def get_encoder_context(self, encoder_turns, num_context):
        # |num_context| utterances before the last partner utterance
        encoder_context = [self._remove_prompt(turn) for turn in encoder_turns[-1*(num_context+1):-1]]
        if len(encoder_context) < num_context:
            batch_size = encoder_turns[0].shape[0]
            empty_context = np.full([batch_size, 1], self.pad, np.int32)
            for i in range(num_context - len(encoder_context)):
                encoder_context.insert(0, empty_context)
        return encoder_context

    def make_decoder_inputs_and_targets(self, decoder_turns, target_turns=None):
        intent = decoder_turns['intent']
        price = [p if p[0] is not None else [0] for p in decoder_turns['price']]
        price_mask = [[1] if p[0] is not None else [0] for p in decoder_turns['original_price']]
        # intent = np.array(intent, dtype=np.int32)
        # price = np.array(price, dtype=np.float)
        # price_mask = np.array(price_mask, dtype=np.int32)
        return intent, price, price_mask

    def _create_one_batch(self, encoder_turns=None, decoder_turns=None,
            target_turns=None, agents=None, uuids=None, kbs=None, kb_context=None,
            price_actions=None,
            num_context=None, encoder_tokens=None, decoder_tokens=None, i=None, roles=None, for_value=False,
            encoder_msgs=None):
        # encoder_context = self.get_encoder_context(encoder_turns, num_context)

        # print('encoder_turns: ', encoder_turns)
        encoder_intent, encoder_price, encoder_price_mask = self.get_encoder_inputs(encoder_turns)
        target_intent, target_price, target_price_mask = self.make_decoder_inputs_and_targets(decoder_turns, target_turns)

        encoder_args = {
                'intent': encoder_intent,
                'price': encoder_price,
                'price_mask': encoder_price_mask,
                'tokens': encoder_tokens,
                }
        extra = [r + [i/self.dia_num] + encoder_price[j][-2:] for j, r in enumerate(roles)]
        encoder_args['extra'] = extra
            # encoder_args['dia_num'] = [i / self.dia_num] * len(encoder_intent)
        # if price_actions['price_act'] is not None:
        #     print('priceactions', price_actions)
        decoder_args = {
                'intent': target_intent,
                'price': target_price,
                'price_mask': target_price_mask,
                'price_act': price_actions['price_act'],
                'prob': price_actions['prob'],
                'context': kb_context,
                }

        context_data = {
                'encoder_tokens': encoder_tokens,
                'decoder_tokens': decoder_tokens,
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                }
        batch = {
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'context_data': context_data,
                }
        return batch

    def int_to_text(self, array, textint_map, stage):
        tokens = [str(x) for x in textint_map.int_to_text((x for x in array if x != self.pad), stage)]
        return ' '.join(tokens)

    def list_to_text(self, tokens):
        return ' '.join(str(x) for x in tokens)

    def print_batch(self, batch, example_id, textint_map, preds=None):
        i = example_id
        print('-------------- Example {} ----------------'.format(example_id))
        if len(batch['decoder_tokens'][i]) == 0:
            print('PADDING')
            return False
        print('RAW INPUT:\n {}'.format(self.list_to_text(batch['encoder_tokens'][i])))
        print('RAW TARGET:\n {}'.format(self.list_to_text(batch['decoder_tokens'][i])))
        print('ENC INPUT:\n {}'.format(self.int_to_text(batch['encoder_args']['intent'][i], textint_map, 'encoding')))
        print('DEC INPUT:\n {}'.format(self.int_to_text(batch['decoder_args']['intent'][i], textint_map, 'decoding')))
        #print('TARGET:\n {}'.format(self.int_to_text(batch['decoder_args']['targets'][i], textint_map, 'target')))
        if preds is not None:
            print('PRED:\n {}'.format(self.int_to_text(preds[i], textint_map, 'target')))
        return True

    def _get_token_turns_at(self, dialogues, i):
        stage = 0
        if not hasattr(dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[i] if i < len(dialogue.token_turns) else ['<pad>']
                for dialogue in dialogues]

    def _get_dialogue_data(self, dialogues):
        '''
        Data at the dialogue level, i.e. same for all turns.
        '''
        agents = self._get_agent_batch_at(dialogues, 1)  # Decoding agent
        kbs = self._get_kb_batch(dialogues)
        uuids = [d.uuid for d in dialogues]
        kb_context_batch = self.create_context_batch(dialogues, self.kb_pad)
        return {
                'agents': agents,
                'kbs': kbs,
                'uuids': uuids,
                'kb_context': kb_context_batch,
                }

    def get_encoding_turn_ids(self, num_turns):
        # NOTE: when creating dialogue turns (see add_utterance), we have set the first utterance to be from the encoding agent
        encode_turn_ids = range(0, num_turns-1, 2)
        return encode_turn_ids

    def _get_lf_batch_at(self, dialogues, i):
        pad = self.mappings['lf_vocab'].to_ind(markers.PAD)
        return pad_list_to_array([d.lfs[i] for d in dialogues], pad, np.int32)

    def create_batch(self, dialogues, for_value=False):
        # num_turns = self._normalize_dialogue(dialogues)
        num_turns = len(dialogues[0].lf_turns)
        dialogue_data = self._get_dialogue_data(dialogues)

        dialogue_class = type(dialogues[0])
        ENC, DEC, TARGET = dialogue_class.ENC, dialogue_class.DEC, dialogue_class.TARGET
        LF, TOKEN, PACT = dialogue_class.LF, dialogue_class.TOKEN, dialogue_class.PACT
        ROLE = dialogue_class.ROLE

        encode_turn_ids = self.get_encoding_turn_ids(num_turns)
        # encoder_turns_all = self._get_turn_batch_at(dialogues, ENC, None)

        # NOTE: encoder_turns contains all previous dialogue context, |num_context|
        # decides how many turns to use
        batch_seq = []
        dias = dialogues
        for i in encode_turn_ids:
            while i+1 >= len(dias[-1].lf_turns):
                dias = dias[:-1]

            batch_seq.append(self._create_one_batch(
                # encoder_turns=encoder_turns_all[:i+1],
                encoder_turns=self._get_turn_batch_at(dias, LF, i, step_back=self.state_length),
                decoder_turns=self._get_turn_batch_at(dias, LF, i+1),
                price_actions=self._get_turn_batch_at(dias, PACT, i+1),
                # target_turns=self._get_turn_batch_at(dialogues, LF, i+1),
                # encoder_tokens=self._get_token_turns_at(dialogues, i),
                encoder_tokens=self._get_turn_batch_at(dias, TOKEN, i),
                # decoder_tokens=self._get_token_turns_at(dialogues, i+1),
                roles=self._get_turn_batch_at(dias, ROLE, i),
                agents=dialogue_data['agents'],
                uuids=dialogue_data['uuids'],
                kbs=dialogue_data['kbs'],
                kb_context=dialogue_data['kb_context'],
                num_context=self.num_context,
                i=i,
                # encoder_msgs=self._get_turn_batch_at(dialogues, dialogue_class.MSG, i),
                ))

        # bath_seq: A sequence of batches that can be processed in turn where
        # the state of each batch is passed on to the next batch

        return batch_seq

class DialogueBatcherWrapper(object):
    def __init__(self, batcher):
        self.batcher = batcher
        # TODO: fix kb_pad, hacky
        self.kb_pad = batcher.kb_pad

    def create_batch(self, dialogues):
        raise NotImplementedError

    def create_context_batch(self, dialogues, pad):
        return self.batcher.create_context_batch(dialogues, pad)

    def get_encoder_inputs(self, encoder_turns):
        return self.batcher.get_encoder_inputs(encoder_turns)

    def get_encoder_context(self, encoder_turns, num_context):
        return self.batcher.get_encoder_context(encoder_turns, num_context)

    def list_to_text(self, tokens):
        return self.batcher.list_to_text(tokens)

    def _get_turn_batch_at(self, dialogues, STAGE, i):
        return self.batcher._get_turn_batch_at(dialogues, STAGE, i)


class DialogueBatcherFactory(object):
    @classmethod
    def get_dialogue_batcher(cls, model, **kwargs):
        if model in ('seq2seq', 'lf2lf', 'tom'):
            batcher = DialogueBatcher(**kwargs)
        else:
            raise ValueError
        return batcher
