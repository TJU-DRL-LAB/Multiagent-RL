import numpy as np
import random
import time

class MinHeap(object):

    def __init__(self, size=20, priority_decay=0.01):
        self._size = size
        self._data = []
        self._count = len(self._data)
        self._priority_decay = priority_decay

    def size(self):
        return self._count

    def __len__(self):
        return self._count

    def __getitem__(self, key):
        if isinstance(key, int):
            assert key < self.size()
            return self._data[key]

    def isEmpty(self):
        return self._count == 0

    def _decay_traj_priority(self):
        for traj in self._data:
            if isinstance(traj, Trajectory):
                traj.priority -= self._priority_decay

    def add(self, item):
        # decay the trajectory's priority with time going by
        self._decay_traj_priority()

        # print("add: ", item.Return)
        if self._count < self._size:
            # 插入元素入堆
            self._data.append(item)
            self._count += 1
            self._shiftup(self._count - 1)
        # 超过给定数量，删除最小
        elif self._count == self._size:
            if len(self._data) == self._size:
                self._data.append(item)
            else:
                self._data[-1] = item
            self._count += 1
            self._shiftup(self._count - 1)
            return self.pop()

    def pop(self):
        # 出堆
        if self._count > 0:
            ret = self._data[0]
            self._data[0] = self._data[self._count - 1]
            self._count -= 1
            self._shiftDown(0)
            return ret

    def peak(self):
        return self._data[0]

    def _shiftup(self, index):
        # 上移self._data[index]，以使它不小于父节点 (只有当新的小于parent的旧的的时候，才有可能互换，相等的时候，保留最新的)
        parent = (index - 1) >> 1
        while index > 0 and self._data[parent] > self._data[index]:
            # swap
            self._data[parent], self._data[index] = self._data[index], self._data[parent]
            index = parent
            parent = (index - 1) >> 1

    def _shiftDown(self, index):
        # 下移self._data[index]，以使它不大于子节点
        j = (index << 1) + 1  # 左节点
        while j < self._count:
            # 有子节点
            if j + 1 < self._count and self._data[j + 1] < self._data[j]:
                # 有右子节点，并且右子节点较小
                j += 1  # 右节点小
            if self._data[index] <= self._data[j]:
                # 堆的索引位置已经大于两个子节点，不需要交换了
                break
            self._data[index], self._data[j] = self._data[j], self._data[index]
            index = j
            j = (index << 1) + 1


class Trajectory(object):
    # min_timestamp = np.inf
    def __init__(self, tuples, Return, priority):
        '''
        用于放入到堆的自定义类。注意要重写__lt__、__ge__、__le__和__cmp__函数。
        '''
        self.trajectory = tuples
        self.Return = Return
        self.priority = priority
        # 目的是新的比旧的大
        self.timestamp = time.time()
        # Trajectory.min_timestamp = min(self.min_timestamp, self.timestamp)

    def __lt__(self, other):
        return self.priority < other.priority or (self.priority == other.priority and self.timestamp < other.timestamp)

    def __gt__(self, other):
        return self.priority > other.priority or (self.priority == other.priority and self.timestamp > other.timestamp)

    def __le__(self, other):
        return self.priority < other.priority or (self.priority == other.priority and self.timestamp < other.timestamp)

    # def __ge__(self, other):
    #     return self.priority >= other.priority

    # def __cmp__(self, other):
    #     # call global(builtin) function cmp for int
    #     return super.cmp(self.priority, other.priority)

    def __str__(self):
        # return '{}, {}'.format(self.trajectory, self.Return)
        return '{}:{}'.format(self.Return, self.timestamp)

    def get(self):
        return self.trajectory, self.Return



class PriorityTrajectoryReplayBuffer(object):
    def __init__(self, size, priority_decay=0.01):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of trajectory to store in the buffer. When the buffer
            overflows the trajectory with the minimum priority are dropped.
        """
        self._storage = MinHeap(size, priority_decay=priority_decay)
        self._maxsize = size
        self.distribution = {}
        self.mean_returns = []
        self.current_mean_return = 0
        print("MinHeap based buffer built...")

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
        self._storage = MinHeap(self._maxsize)

    def add(self, trajectory, R):
        # distribution
        self.add_item(R)

        data = Trajectory(trajectory, R, R)
        current_sum = len(self._storage) * self.current_mean_return
        current_sum += R

        removed_item = self._storage.add(data)

        if removed_item is not None:
            # print("remove: ", removed_item.Return)
            self.remove_item(removed_item.Return)
            current_sum -= removed_item.Return

        self.current_mean_return = current_sum / len(self._storage)
        self.mean_returns.append(self.current_mean_return)

    def _encode_sample(self, idxes, weights=None):
        '''
        sample trajectories
        :param idxes: trajectory index
        :return:
        '''
        # trajectory_k = []
        tra_obses_t, tra_actions, tra_rewards, tra_obses_tp1, tra_dones, tra_returns, tra_dis_2_ends, tuple_weights, ranges = [], [], [], [], [], [], [], [], []

        trajectory_idx, tuple_idx, weight = 0, 0, 1
        for i in idxes:
            traj_obj = self._storage[i]  # sample a trajectory
            trajectory, Return = traj_obj.get()
            if weights is not None:
                weight = weights[trajectory_idx]
            for (obs_t, action, reward, obs_tp1, done, dis_2_end, R) in trajectory:
                tra_obses_t.append(np.array(obs_t, copy=False))
                tra_actions.append(np.array(action, copy=False))
                tra_rewards.append(reward)
                tra_obses_tp1.append(np.array(obs_tp1, copy=False))
                tra_dones.append(done)
                tra_dis_2_ends.append(dis_2_end)
                tra_returns.append(R)
                tuple_weights.append(weight)
            tuple_idx += len(trajectory)
            ranges.append(tuple_idx)
            trajectory_idx += 1
        return np.array(tra_obses_t), np.array(tra_actions), np.array(tra_rewards), np.array(tra_obses_tp1), \
               np.array(tra_dones), np.array(tra_dis_2_ends), np.array(tra_returns), np.array(tuple_weights), ranges
        # np.random.shuffle(idx)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

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


def test_min_queue():
    queue = MinHeap(20)
    iLen = random.randint(1, 300)
    allData = random.sample(range(iLen * 100), iLen)
    for i in allData:
        queue.add(i)
    print(len(queue))
    # print(queue._data)
    data = [queue.pop() for i in range(queue._size)]
    print(data)

def test_priority_buffer():
    from prioritized_experience_replay_buffer.utils import reload_data
    buffer = PriorityTrajectoryReplayBuffer(5)
    positive_buffer = reload_data('D:\program/Python/conference/aaai_19_gan/experiments/2v3_climbing/demonstrations/3000/climbing_hard_demonstration_buffers.pkl')[0]
    trajectory, tra_length = [], []
    count = 0
    for item in positive_buffer._storage:
        obs_t, action, reward, obs_tp1, done, dis_2_end, R = item
        trajectory.append([obs_t, action, reward, obs_tp1, done, dis_2_end, R])
        if int(done) == 1:
            count += 1
            Return = R
            print("return: ", Return)
            buffer.add(trajectory, Return)
            tra_length.append(len(trajectory))
            trajectory = []
    print("Total {} experts' trajectories, mean length is {}.".format(count, np.mean(tra_length)))
    return buffer

if __name__ == '__main__':
    buffer = test_priority_buffer()
    print(buffer.make_index(7))
    data = [str(buffer._storage.pop()) for i in range(len(buffer))]
    print(data)
    print(Trajectory.min_timestamp)
    # print(buffer.sample(5))
