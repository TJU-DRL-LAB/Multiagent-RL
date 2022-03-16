class AgentTrainer(object):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    def get_actions(self, obs):
        raise NotImplemented()

    def experience(self, obs, act, rew, new_obs, done):
        raise NotImplemented()

    def preupdate(self):
        raise NotImplemented()

    def do_training(self, agents, iter, episode):
        raise NotImplemented()


from algorithm.trainer.ddpg.maddpg import MADDPGAgentTrainer
from algorithm.trainer.simple.fixed import FixedPrey
from algorithm.trainer.simple.random_agent import RandomAgent
from algorithm.trainer.gasil.gasil import GASIL_DDPGAgentTrainer
from algorithm.prioritized_experience_replay_buffer.replay_buffer import ReplayBuffer as NormalReplayBuffer
from algorithm.prioritized_experience_replay_buffer.replay_buffer import PrioritizedReplayBuffer as PrioritizedReplayBuffer
from algorithm.prioritized_experience_replay_buffer.priority_queue_buffer import PriorityTrajectoryReplayBuffer
from algorithm.prioritized_experience_replay_buffer.trajectory_replay_buffer import TrajectoryReplayBuffer
from algorithm.common.network_utils import mlp_model, discriminator



class SimpleAgentFactory(object):

    @staticmethod
    def createAgent(agent_idx, policy_name, obs_shape_n, action_dim_n, hyper_parameters):
        if policy_name == 'ddpg' or policy_name == 'maddpg':
            if hyper_parameters.prioritized_er:
                pool = PrioritizedReplayBuffer(hyper_parameters.buffer_size, alpha=hyper_parameters.alpha)
            else:
                pool = NormalReplayBuffer(hyper_parameters.buffer_size, save_return=hyper_parameters.save_return)
            agent = MADDPGAgentTrainer(
                "agent_%d" % agent_idx, mlp_model, obs_shape_n, action_dim_n, agent_idx,
                buffer=pool, local_q_func=(policy_name.find('maddpg') == -1))
            print("Agent {} uses uniform experience replay buffer...".format(agent_idx))
        elif policy_name == 'fixed':
            agent = FixedPrey(agent_idx, action_dim_n[agent_idx])
            print("Agent {} is fixed...".format(agent_idx))
        elif policy_name == 'random':
            agent = RandomAgent(agent_idx, action_dim_n[agent_idx])
            print("Agent {} is random...".format(agent_idx))
        elif policy_name == 'gasil':
            print("Agent {} is self imitation...".format(agent_idx))
            # positive pool to save positive experience
            positive_pool = PriorityTrajectoryReplayBuffer(hyper_parameters.positive_buffer_size)
            # save recent trajectory
            trajectory_pool = TrajectoryReplayBuffer(hyper_parameters.positive_buffer_size)
            # normal buffer for policy gradient
            pool = NormalReplayBuffer(hyper_parameters.buffer_size, save_return=hyper_parameters.save_return)

            agent = GASIL_DDPGAgentTrainer(
                "agent_%d" % agent_idx, mlp_model, discriminator, obs_shape_n, action_dim_n, agent_idx,
                buffer=pool, trajectory_buffer=trajectory_pool, positive_buffer=positive_pool,
                local_q_func=True)
            print(
                "Agent {} uses a uniform experience replay buffer a unifrom trajectory experience buffer and a prioritized (Minimum Heap based) trajectory experience replay buffer...".format(
                    agent_idx))
            return agent
        else:
            agent = FixedPrey(agent_idx, action_dim_n[agent_idx])

        return agent