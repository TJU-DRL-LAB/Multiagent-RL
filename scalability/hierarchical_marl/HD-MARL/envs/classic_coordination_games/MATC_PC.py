#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc

# SEED = 222
# np.random.seed(sd.SEED)

class AgentObj:
    def __init__(self, coordinates, type, name, direction=0, mark=0, hidden=0):
        self.x = coordinates[0]
        self.y = coordinates[1]
        #0: r, 1: g, 3: b
        self.type = type
        self.name = name
        self.hidden = hidden

        # 0: right, 1:top 2: left. 3: bottom
        # self.direction = direction
        self.mark = mark

        # 0: empty, [1,n]: trash i
        # self.load = 0
        self.load_status = None
        self.action_space = ['forward', 'backward', 'left', 'right', 'stay', 'pick', 'place']

        self.reward = 0

    def move_delta(self, action):
        if action == 0:
            delta_x, delta_y = 0, 1
        elif action == 1:
            delta_x, delta_y = 0, -1
        elif action == 2:
            delta_x, delta_y = -1, 0
        elif action == 3:
            delta_x, delta_y = 1, 0
        elif action == 4:
            delta_x, delta_y = 0, 0
        else:
            assert action in range(5), 'wrong direction'

        return delta_x, delta_y

    def move(self, action, env_x_size, env_y_size):
        delta_x, delta_y = self.move_delta(action)
        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y

        if not self.load_status is None:
            self.load_status.coordinates_update(self.x, self.y)

        return self.x, self.y

    def load(self, trash):
        if trash is None:
            return
        assert self.load_status is None, "currently loading."
        self.load_status = trash
        trash.loaded(self.mark)

    def place(self):
        assert self.load_status is not None, "not loading."
        trash = self.load_status
        trash.placed()
        self.load_status = None
        return trash

    def get_coordinate(self):
        return self.x, self.y

    def get_load(self):
        return self.load_status

    def is_loading(self):
        return self.load_status is not None

    def get_reward(self, r=0):
        self.reward += r
        return self.reward



class TrashObj:
    def __init__(self, coordinates, mark, type=1, hidden=0, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type

        self.hidden = hidden
        self.reward = reward

        self.mark = mark
        # 0: normal, [1,n]: loaded by robo i, -1: dumped
        self.status = 0

    def get_status(self):
        return self.status

    def is_loaded(self):
        return self.status > 0

    def loaded(self, i):
        assert self.status == 0, 'not normal state -- 0'
        self.status = i

    def placed(self):
        assert self.status > 0, 'not normal state -- i'
        self.status = 0

    def dumped(self):
        assert self.status == 0, 'not normal state -- 0'
        self.status = -1

    def coordinates_update(self, x, y):
        self.x = x
        self.y = y
        return self.get_coordinate()

    def get_coordinate(self):
        return self.x, self.y


class DumpObj:
    def __init__(self, coordinates, mark):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.dump = []

        self.mark = mark

    def receive_trash(self, trash):
        trash.dumped()
        self.dump.append(trash.mark)

    def get_dump(self):
        return self.dump

    def get_coordinate(self):
        return self.x, self.y

class GameEnv:
    def __init__(self, width=15, height=7, agent_num=3, trash_num=2, render_size=84, render_type=1, is_fixed=True):
        self.size_x = width
        self.size_y = height
        self.objects = []

        self.agent_num = agent_num
        self.trash_num = trash_num

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: stay, 5: load, 6: place
        self.action_num = 8

        # 0: dump 1, 1: dump 2, 2: dump 3, 3: trash 1/2, 4: load, 5: place
        self.goal_num = 5
        self.render_size = render_size
        # 0 for grey, 1 for rgb
        self.render_type = render_type

        self.is_fixed = is_fixed
        # self.done = False
        self.reset()

    def reset(self):
        # init dumps, A (10,1), B (10,10)
        self.dump1 = DumpObj(coordinates=(7, 1), mark=1)
        self.dump2 = DumpObj(coordinates=(7, 5), mark=2)

        # init agents, A (3,4), B (3,6)
        self.agent1 = AgentObj(coordinates=(3, 3), type=2, mark=1, name='agent1')
        self.agent2 = AgentObj(coordinates=(10, 3), type=0, mark=2, name='agent2')

        # init trashs, A (1,4), B (0,6)
        self.trash_objects = []
        self.trash1 = TrashObj(coordinates=(0, 3), mark=1)
        self.trash2 = TrashObj(coordinates=(14, 3), mark=2)

        # init walls
        self.walls = [(7, 0), (7, 2), (7, 3), (7, 4), (7, 6)]

        self.trash_objects.append(self.trash1)
        self.trash_objects.append(self.trash2)

        s = self.train_render()
        return s


    def step(self, agent1_action, agent2_action, agent1_goal, agent2_goal, nom=False):
        assert agent1_action in range(self.action_num), 'agent1 takes wrong action'
        assert agent2_action in range(self.action_num), 'agent2 takes wrong action'

        assert agent1_goal in range(self.goal_num), 'agent1 takes illegal goal'
        assert agent2_goal in range(self.goal_num), 'agent2 takes illegal goal'

        agent1_old_x, agent1_old_y = self.agent1.x, self.agent1.y
        agent2_old_x, agent2_old_y = self.agent2.x, self.agent2.y

        r1 = 0.0
        r2 = 0.0
        done = False
        ir1 = -0.01
        ir2 = -0.01
        reached1 = False
        reached2 = False

        # move-actions
        if agent1_action in range(5):
            agent1_action_return = self.agent1.move(agent1_action, env_x_size=self.size_x, env_y_size=self.size_y)
        if agent2_action in range(5):
            agent2_action_return = self.agent2.move(agent2_action, env_x_size=self.size_x, env_y_size=self.size_y)

        # TODO no step in walls
        if (self.agent1.x, self.agent1.y) in self.walls:
            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            # TODO no collision on walls
            # ir1 = -0.1
        if (self.agent2.x, self.agent2.y) in self.walls:
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y
            # TODO no collision on walls
            # ir2 = -0.1

        # TODO no step in dumps
        if (self.agent1.x, self.agent1.y) == (self.dump1.x, self.dump1.y) \
                or (self.agent1.x, self.agent1.y) == (self.dump2.x, self.dump2.y):
            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
        if (self.agent2.x, self.agent2.y) == (self.dump1.x, self.dump1.y) \
                or (self.agent2.x, self.agent2.y) == (self.dump2.x, self.dump2.y):
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        # TODO collision between agents
        if self.agent1.x == agent2_old_x and self.agent1.y == agent2_old_y and \
                        self.agent2.x == agent1_old_x and self.agent2.y == agent1_old_y:
            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        if self.agent1.x == self.agent2.x and self.agent1.y == self.agent2.y:
            # TODO
            # if np.random.random() < 0.5:
            #     self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            # else:
            #     self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y
            # agent1/agent2 intrude the other one.
            if (self.agent1.x == agent1_old_x and self.agent1.y == agent1_old_y) or \
                (self.agent2.x == agent2_old_x and self.agent2.y == agent2_old_y):
                self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
                self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y
            else:
                if np.random.random() < 0.5:
                    self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
                else:
                    self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        # TODO
        reached1 = self.checkIfReachGoal(self.agent1.x, self.agent1.y, agent1_goal)
        if reached1:
            ir1 = 1

        reached2 = self.checkIfReachGoal(self.agent2.x, self.agent2.y, agent2_goal)
        if reached2:
            ir2 = 1

        # load
        if agent1_action == 5:
            if self.agent1.load_status is None:
                for trash in self.trash_objects:
                    if trash.get_status() == 0 and trash.x == self.agent1.x and trash.y == self.agent1.y:
                        # print('agent 1 load trash', trash.mark)
                        self.agent1.load(trash)
                        # r1 = 0.0
                        break
            # else:
            #     r1 = -0.1
                # r1 = -1

        if agent2_action == 5:
            if self.agent2.load_status is None:
                for trash in self.trash_objects:
                    if trash.get_status() == 0 and trash.x == self.agent2.x and trash.y == self.agent2.y:
                        # print('agent 2 load trash', trash.mark)
                        self.agent2.load(trash)
                        # r2 = 0.0
                        break
            # else:
            #     r2 = -0.1
                # r2 = -1

        # TODO reward = 1 -> 0
        # place
        if agent1_action == 6:
            if self.agent1.is_loading():
                # print('agent 1 place trash', self.agent1.load_status.mark)
                trash = self.agent1.place()
                if self.dump1.x - 1 == self.agent1.x and self.dump1.y == self.agent1.y:
                    self.dump1.receive_trash(trash)
                    # r1 = 1.0
                if self.dump2.x - 1 == self.agent1.x and self.dump2.y == self.agent1.y:
                    self.dump2.receive_trash(trash)
                    # r1 = 1.0
            # else:
            #     r1 = -0.1
                # r1 = -1

        if agent2_action == 6:
            if self.agent2.is_loading():
                # print('agent 2 place trash', self.agent2.load_status.mark)
                trash = self.agent2.place()
                if self.dump1.x + 1 == self.agent2.x and self.dump1.y == self.agent2.y:
                    self.dump1.receive_trash(trash)
                    # r2 = 1.0
                if self.dump2.x + 1 == self.agent2.x and self.dump2.y == self.agent2.y:
                    self.dump2.receive_trash(trash)
                    # r2 = 1.0
            # else:
            #     r2 = -0.1
                # r2 = -1

        # terminate condition
        trash_collected = self.dump1.dump + self.dump2.dump
        if len(trash_collected) == 2:
            r1, r2 = 0.1, 0.1
            if len(self.dump1.dump) == 2:
                r1, r2 = 1.0, 1.0
            if len(self.dump2.dump) == 2:
                r1, r2 = 0.5, 0.5
            # r1 += 1.0
            # r2 += 1.0
            done = True

        s_ = self.train_render()

        m1_ = 0 if self.agent1.load_status is None else 1
        m2_ = 0 if self.agent2.load_status is None else 1

        return s_, m1_, m2_, ir1, ir2, r1, r2, done, reached1, reached2


    def contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 4], dtype=np.float32)
        a[1:-1, 1:-1, :] = 0

        a[self.dump1.y + 1, self.dump1.x + 1, 2] = 1
        a[self.dump2.y + 1, self.dump2.x + 1, 2] = 1

        for trash in self.trash_objects:
            if trash.get_status() == 0:
                a[trash.y + 1, trash.x + 1, 1] = 1

        a[self.agent1.y + 1, self.agent1.x + 1, 0] = 1
        a[self.agent2.y + 1, self.agent2.x + 1, 0] = 1

        # walls info
        for (x, y) in self.walls:
            a[y + 1, x + 1, 3] = 1

        return a

    def render_contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3], dtype=np.float32)
        a[1:-1, 1:-1, :] = 0

        for i in range(2):
            a[self.dump1.y + 1, self.dump1.x + 1, i] = 1
            a[self.dump2.y + 1, self.dump2.x + 1, i] = 1

        for trash in self.trash_objects:
            if trash.get_status() == 0:
                for i in range(3):
                    a[trash.y + 1, trash.x + 1, i] = 1 if i == trash.type else 0
            # TODO render the dumped trash position
            elif trash.get_status() == -1:
                a[trash.y + 1, trash.x + 1, 1] = 1 if i == trash.type else 0
                a[trash.y + 1, trash.x + 1, 2] = 1 if i == trash.type else 0

        for i in range(3):
            # delta_x, delta_y = self.agent1.move_forward_delta()
            # a[self.agent1.y + 1 + delta_y, self.agent1.x + 1 + delta_x, i] = 0.5
            # if self.agent1.is_loading():
            #     a[self.agent1.y + 1 - delta_y, self.agent1.x + 1 - delta_x, i] = 1 if i == 1 else 0
            #
            # delta_x, delta_y = self.agent2.move_forward_delta()
            # a[self.agent2.y + 1 + delta_y, self.agent2.x + 1 + delta_x, i] = 0.5
            # if self.agent2.is_loading():
            #     a[self.agent2.y + 1 - delta_y, self.agent2.x + 1 - delta_x, i] = 1 if i == 1 else 0

            a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent1.type else 0
            a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent2.type else 0

        for i in range(3):
            for (x, y) in self.walls:
                a[y + 1, x + 1, i] = 0.5

        # print(a[:,:,1])
        return a

    def render_env(self):
        a = self.render_contribute_metrix()

        b = scipy.misc.imresize(a[:, :, 0], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')

        # b = scipy.misc.imresize(a[:, :, 0], [self.render_size, self.render_size, 1], interp='nearest')
        # c = scipy.misc.imresize(a[:, :, 1], [self.render_size, self.render_size, 1], interp='nearest')
        # d = scipy.misc.imresize(a[:, :, 2], [self.render_size, self.render_size, 1], interp='nearest')

        if self.render_type == 0:
            e = b * 0.299 + c * 0.587 + d * 0.114
            # e = np.array(e, dtype=np.float32).reshape([5 * self.size_y, 5 * self.size_x, 1])
            e = np.stack([e, e, e], axis=2)
        else:
            e = np.stack([b, c, d], axis=2)

        return e

    # TODO return state in different order for diff agents
    def train_render(self):
        a = self.contribute_metrix()
        # a_trim = a[1:-1, 1:-1, :]
        # s1 = a_trim[:, :, 0].flatten().tolist()
        # s2 = a_trim[:, :, 1].flatten().tolist()
        # e = a_trim[:, :, 2:].flatten().tolist()
        # s = []
        # s.append(s1 + s2 + e)
        # s.append(s2 + s1 + e)
        e = a[1:-1, 1:-1, :].flatten()
        return e.tolist()

    # def encodeImage(self, a):
    #     b = scipy.misc.imresize(a[:, :, 0], [self.render_size, self.render_size, 1], interp='nearest')
    #     c = scipy.misc.imresize(a[:, :, 1], [self.render_size, self.render_size, 1], interp='nearest')
    #     d = scipy.misc.imresize(a[:, :, 2], [self.render_size, self.render_size, 1], interp='nearest')
    #
    #     if self.render_type == 0:
    #         e = b * 0.299 + c * 0.587 + d * 0.114
    #         e = np.array(e, dtype=np.float32).reshape([self.render_size, self.render_size, 1])
    #     else:
    #         e = np.stack([b, c, d], axis=2)
    #     return e

    # TODO bug fixed
    def encodeGoal(self, goal, agent):
        a = np.ones([self.size_y + 2, self.size_x + 2], dtype=np.float32)
        a[1:-1, 1:-1] = 0

        off_set = -1 if agent == 0 else 1
        if goal == 0:
            a[self.dump1.y + 1, self.dump1.x + 1 + off_set] = 1

        if goal == 1:
            a[self.dump2.y + 1, self.dump2.x + 1 + off_set] = 1

        if goal == 2:
            if agent == 0:
                if self.trash1.get_status() == 0:
                    a[self.trash1.y + 1, self.trash1.x + 1] = 1
            if agent == 1:
                if self.trash2.get_status() == 0:
                    a[self.trash2.y + 1, self.trash2.x + 1] = 1

        e = a[1:-1, 1:-1].flatten()

        return e.tolist()

    def checkIfReachGoal(self, x, y, g):
        reach = False
        if g == 0:
            if (x, y) == (self.dump1.x + 1, self.dump1.y) or (x, y) == (self.dump1.x - 1, self.dump1.y):
                reach = True
        elif g == 1:
            if (x, y) == (self.dump2.x + 1, self.dump2.y) or (x, y) == (self.dump2.x - 1, self.dump2.y):
                reach = True
        elif g == 2:
            if (x, y) == (self.trash1.x, self.trash1.y) or (x, y) == (self.trash2.x, self.trash2.y):
                reach = True

        return reach
