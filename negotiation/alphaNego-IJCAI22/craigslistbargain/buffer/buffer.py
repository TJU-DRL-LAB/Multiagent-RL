from neural.batcher_rl import RawBatch

import numpy as np
import torch
import random


class ReplayBuffer:
    """
    Special Buffer
    """
    def __init__(self, max_size=2560, batch_type=RawBatch,
                 default_device='cpu'):

        self.max_size = max_size
        self._top = 0
        self._cur = 0
        self.batch_type = batch_type
        self.buffer = [None]*max_size
        self.batch_lengths = np.zeros(max_size, dtype=np.int32)
        self.default_device = default_device

    _buffer_instances = {}

    @staticmethod
    def global_init(name, **kwargs):
        ReplayBuffer._buffer_instances[name] = ReplayBuffer(**kwargs)

    @staticmethod
    def get_instance(name):
        return ReplayBuffer._buffer_instances[name]

    @property
    def size(self):
        return self._top

    def empty(self):
        self._top, self._cur = 0, 0
        self.buffer = [None]*self.max_size

    def sample_batch(self, batch_size, start_from=-1,
                     to_device=None, batch_type=None, sample_mode='unique',
                     add_info=None):
        # sample data batch and move to one storage
        if batch_type is None:
            batch_type = self.batch_type
        # start_from == -1: randomly sample
        if self.size < batch_size:
            batch_size = self.size

        if sample_mode == "unique":
            indices = random.sample(range(self.size), batch_size)
        else:
            indices = np.random.randint(0, self.size, batch_size)
        biters = [self.buffer[i] for i in indices]
        lengths = [self.batch_lengths[i] for i in indices]

        biters, sorted_id = self._sort_cat_iters(biters, lengths)
        sorted_id = [indices[i] for i in sorted_id]

        biters = [self._dicts_to_batches(bi, batch_type, to_device) for bi in biters]

        # Additional information
        if add_info is None:
            add_info = []
        ret_add = {}
        for k in add_info:
            kk = '_'+k
            ret_add[k] = [getattr(self, kk)[i] for i in sorted_id]
        return biters, sorted_id, ret_add

    def add_batch_iters(self, batch_iters, add_dict=None):
        # added one by one
        if add_dict is None:
            add_dict = {}
        for i, bi in enumerate(batch_iters):
            tmp_dict = {k: add_dict[k][i] for k in add_dict}
            self.add_batch_iter(batch_iter=bi, add_dict=tmp_dict)

    def add_batch_iter(self, batch_iter, add_dict):
        # Transfer batch to dict and storage in cpu memory
        if self._top < self.max_size:
            self._top += 1
        _cur = self._cur
        self._cur += 1
        if self._cur == self.max_size:
            self._cur = 0

        # Lock free below
        self.buffer[_cur], self.batch_lengths[_cur] = self._batches_to_dicts(batch_iter, self.default_device)
        for k in add_dict:
            kk = '_'+k
            if not hasattr(self, kk):
                setattr(self, kk, [None]*self.max_size)
            getattr(self, kk)[_cur] = add_dict[k]

    def _sort_iters(self, biters, length):
        sorted_id = sorted([i for i in range(len(biters))], reverse=True, key=lambda x: len(biters[x]))
        newb = [biters[i] for i in sorted_id]
        # biters = sorted(biters, reverse=True, key=lambda l: len(l))
        length = [len(bi) for bi in newb]
        return newb, length, sorted_id

    def _cat_iters(self, biters, length):
        rets = []
        # for each column
        for l in range(len(biters[0])):
            ret = {}
            # for each elements
            for k in biters[0][0]:
                es = []
                torch_cat = False
                # for each row
                for bi in biters:
                    if len(bi) <= l:
                        break
                    if isinstance(bi[l][k], list):
                        es.extend(bi[l][k])
                    else:
                        es.append(bi[l][k])
                        torch_cat = True
                if torch_cat:
                    es = torch.cat(es, dim=0)
                ret[k] = es
            rets.append(ret)
        return rets

    def _sort_cat_iters(self, biters, length, sub_size=-1):
        if sub_size == -1:
            sub_size = len(biters)
        biters, length, sorted_id = self._sort_iters(biters, length)
        cur = 0
        new_biters = []
        batch_size = len(biters)
        while cur < batch_size:
            step = sub_size
            if cur + step > batch_size:
                step = batch_size-cur
            tmp = self._cat_iters(biters[cur: cur+step], length[cur:cur+step])
            new_biters.append(tmp)
            cur += step

        return new_biters, sorted_id

    def _process_element(self, e, to_device=None):
        if isinstance(e, tuple) or isinstance(e, list):
            if isinstance(e, int) or isinstance(e, float):
                return e
            return type(e)(self._process_element(ee, to_device) for ee in e)
        elif isinstance(e, torch.Tensor):
            if to_device is None:
                return e
            return e.to(device=to_device)
        else:
            return e

    def _dicts_to_batches(self, bs, batch_type, to_device):
        ret = []
        # for each column
        for b in bs:
            bb = b.copy()
            for k in bb:
                bb[k] = self._process_element(bb[k], to_device)
            ret.append(batch_type(**bb))

        return ret

    def _batches_to_dicts(self, bs, to_device):
        # Transfer a batch of one dialogue to dict list
        ret = []
        for b in bs:
            d = {}
            for k in b.tensor_attributes:
                tmp = getattr(b, k)
                d[k] = self._process_element(tmp, to_device=to_device)
            ret.append(d)
        return ret, len(ret)