from algorithm.prioritized_experience_replay_buffer.replay_buffer import ReplayBuffer
import numpy as np
class RandomAgent():
    def __init__(self, agent_idx, action_dim):
        self.agent_idx = agent_idx
        self.action_dim = action_dim

        self.pool = ReplayBuffer(1e6)
        print("random action_dim: ", action_dim)

    def get_actions(self, observations, single=True):
        # [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
        return [np.random.rand() * 2 - 1 for _ in range(self.action_dim)]

    # def target_action(self, observations):
    #     return [[np.random.rand() * 2 - 1 for _ in range(self.action_dim)]] * observations.shape[0]

    def experience(self, obs, action, reward, obs_nxt, done):
        self.pool.add(obs, action, reward, obs_nxt, float(done), None, None)

    def _do_training(self, agents, iteration, batch_size, epoch_len):
        pass

    def is_exploration_enough(self, min_pool_size):
        return False