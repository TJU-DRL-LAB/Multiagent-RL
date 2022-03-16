from algorithm.trainer import AgentTrainer

class FixedPrey(AgentTrainer):
    def __init__(self, agent_idx, action_dim):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        print("random action_dim: ", action_dim)

    def get_actions(self, observations, single=True):
        return [0 for _ in range(self.action_dim)]