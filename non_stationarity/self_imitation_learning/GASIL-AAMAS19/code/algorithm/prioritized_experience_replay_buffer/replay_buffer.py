import numpy as np
import random
from algorithm.prioritized_experience_replay_buffer.segment_tree import MinSegmentTree
from algorithm.prioritized_experience_replay_buffer.segment_tree import SumSegmentTree

class ReplayBuffer(object):
    def __init__(self, size, save_return=False):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.save_return = save_return
        self.distribution = {}
        self.mean_returns = []
        self.current_mean_return = 0

    def __len__(self):
        return len(self._storage)

    def add_item(self, R):
        if self.distribution.get(R, -1) == -1:
            self.distribution[R] = 1
        else:
            self.distribution[R] += 1

    def remove_item(self, R):
        # old = self.distribution[R]
        self.distribution[R] -= 1

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done, dis_2_end, R=None):
        if self.save_return:
            data = (obs_t, action, reward, obs_tp1, done, dis_2_end, R)
            # distribution
            self.add_item(R)
            current_sum = len(self._storage) * self.current_mean_return
            current_sum += R
        else:
            data = (obs_t, action, reward, obs_tp1, done, dis_2_end)


        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            if self.save_return:
                # if add return, remove from distribution
                self.remove_item(self._storage[self._next_idx][-1])
                current_sum -= self._storage[self._next_idx][-1]
            self._storage[self._next_idx] = data
        if self.save_return:
            self.current_mean_return = current_sum / len(self._storage)
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, idxes):
        if self.save_return:
            obses_t, actions, rewards, obses_tp1, dones, returns, dis_2_ends = [], [], [], [], [], [], []
            for i in idxes:
                data = self._storage[i]
                obs_t, action, reward, obs_tp1, done, dis_2_end, R = data

                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                dis_2_ends.append(dis_2_end)
                returns.append(R)
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(
                dones), np.array(dis_2_ends), np.array(returns)
        else:
            obses_t, actions, rewards, obses_tp1, dones, dis_2_ends = [], [], [], [], [], []
            for i in idxes:
                data = self._storage[i]
                obs_t, action, reward, obs_tp1, done, dis_2_end = data

                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                dis_2_ends.append(dis_2_end)
            return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(
                dones), np.array(dis_2_ends), None

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, save_return=True)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def make_index(self, batch_size):
        return self._sample_proportional(batch_size=batch_size)

    def sample_index(self, idxes, beta=0.4):
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                # # print('p_sample: ', p_sample)
                # # print('size: ', len(self._storage))
                # try:
                #     weight = (p_sample * len(self._storage)) ** (-beta)
                # except ZeroDivisionError as e:
                #     print(self.distribution)
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        R_batch: np.array
            returns received as results of executing act_batch
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)

        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

