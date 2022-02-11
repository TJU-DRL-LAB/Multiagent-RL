import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNNMaxRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        print('use gnn + max + rnn agent')
        super(GNNMaxRnnAgent, self).__init__()
        self.args = args

        # feature index
        self.move_feat_end = args.move_feats_size

        self.enemies_feat_start = args.move_feats_size

        self.enemies_num = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num

        self.agents_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + 1

        self.agents_num = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + 1 + args.agent_feats_size * (self.args.agents_num - 1)

        self.blood_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + 1 + args.agent_feats_size * (self.args.agents_num - 1) + 1

        self.blood_feat_end = self.blood_feat_start + 1

        self.other_feat_start = args.move_feats_size + args.enemy_feats_size * self.args.enemies_num + 1 + args.agent_feats_size * (self.args.agents_num - 1) + 1 + 1

        # network struct
        self.self_info_fc1 = nn.Linear(5, args.arn_hidden_size)

        self.enemies_info_fc1 = nn.Linear(args.enemy_feats_size, args.arn_hidden_size, bias=False)

        self.agents_info_fc1 = nn.Linear(args.agent_feats_size, args.arn_hidden_size, bias=False)

        self.rnn = nn.GRUCell(args.arn_hidden_size * 3, args.arn_hidden_size)

        self.action_ouput_fc = nn.Linear(args.arn_hidden_size,  args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.self_info_fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        self_input = th.cat([inputs[:, :self.move_feat_end],
                             inputs[:, self.blood_feat_start: self.blood_feat_end]],
                            dim=1)

        agents_feats = th.stack([inputs[:, self.agents_feat_start + i * self.args.agent_feats_size:
                                           self.agents_feat_start + self.args.agent_feats_size * (1 + i)]
                         for i in range(self.args.agents_num - 1)], dim=1)

        enemies_feats = th.stack([inputs[:, self.enemies_feat_start + i * self.args.enemy_feats_size:
                                            self.enemies_feat_start + self.args.enemy_feats_size * (1 + i)]
                         for i in range(self.args.enemies_num)], dim=1)

        agent_nums = inputs[:, self.agents_num].reshape(-1).int()
        enemy_nums = inputs[:, self.enemies_num].reshape(-1).int()

        self_hidden = F.relu(self.self_info_fc1(self_input))

        # 因为看不到+死掉的agent的feat均为0，同时关掉了bias，看不到的+死掉的agent的feat过完网络之后，还是都是0
        # 0是relu之后的最小值，所以求max即可
        agents_hidden = F.relu(self.agents_info_fc1(agents_feats))
        max_agents_hidden = th.max(agents_hidden, dim=1)[0]

        enemies_hidden = F.relu(self.enemies_info_fc1(enemies_feats))
        max_enemies_hidden = th.max(enemies_hidden, dim=1)[0]


        # print('enemy:', enemy_nums)
        # print(max_enemies_hidden)

        all_feat = th.cat([self_hidden, max_agents_hidden, max_enemies_hidden], dim=-1)

        h_in = hidden_state.reshape(-1, self.args.arn_hidden_size)
        hidden_state = self.rnn(all_feat, h_in)

        q = self.action_ouput_fc(hidden_state)

        # if self.args.save_agent:
        #     for idx, value in enumerate(list(agent_nums.detach().cpu().numpy())):
        #         if value == self.args.see_num:
        #             print(max_agents_hidden.detach().cpu().numpy().tolist()[idx])
        #
        # if not self.args.save_agent:
        #     for idx, value in enumerate(list(enemy_nums.detach().cpu().numpy())):
        #         if value == self.args.see_num:
        #             print(max_enemies_hidden.detach().cpu().numpy().tolist()[idx])

        return q, hidden_state
