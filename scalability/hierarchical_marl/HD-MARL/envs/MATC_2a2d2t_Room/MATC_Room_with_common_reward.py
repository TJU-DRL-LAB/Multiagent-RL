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


    # def turn_left(self, **kwargs):
    #     self.direction = (self.direction + 1) % 4
    #     return self.direction
    #
    # def turn_right(self, **kwargs):
    #     self.direction = (self.direction - 1 + 4) % 4
    #     return self.direction

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
    def __init__(self, width=11, height=11, agent_num=3, trash_num=2, render_size=84, render_type=1, is_fixed=True):
        self.size_x = width
        self.size_y = height
        self.objects = []

        self.agent_num = agent_num
        self.trash_num = trash_num

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: stay, 5: load, 6: place
        self.action_num = 7

        # 0: dump 1, 1: dump 2, 2: trash 1, 3: trash 2, 4: load, 5: place
        self.goal_num = 6
        self.render_size = render_size
        # 0 for grey, 1 for rgb
        self.render_type = render_type

        self.is_fixed = is_fixed
        # self.done = False
        self.reset()

    def reset(self):
        # init dumps, A (10,1), B (10,10)
        self.dump1 = DumpObj(coordinates=(self.size_x - 1, 1), mark=1)
        self.dump2 = DumpObj(coordinates=(self.size_x - 1, self.size_y - 1), mark=2)

        # init agents, A (3,4), B (3,6)
        self.agent1 = AgentObj(coordinates=(3, int((self.size_y - 1)/2) - 1), type=2, mark=1, name='agent1')
        self.agent2 = AgentObj(coordinates=(3, int((self.size_y - 1)/2) + 1), type=0, mark=2, name='agent2')

        # init trashs, A (1,4), B (0,6)
        self.trash_objects = []
        self.trash1 = TrashObj(coordinates=(1, int((self.size_y - 1)/2) - 1), mark=1)
        self.trash2 = TrashObj(coordinates=(0, int((self.size_y - 1)/2) + 1), mark=2)

        # init walls
        self.walls = [(0, int((self.size_y - 1)/2)), (1, int((self.size_y - 1)/2)), (2, int((self.size_y - 1)/2))]
        self.walls += [(2, i) for i in range(2, self.size_y - 2)]
        self.walls += [(self.size_x - 3, i) for i in range(self.size_y - 4, self.size_y)]
        self.walls += [(i, 2) for i in range(self.size_x - 4, self.size_x)]
        self.walls += [(self.size_x - 4, 1)]

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

        r = 0.0
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
        if (self.agent2.x, self.agent2.y) in self.walls:
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y
            # TODO no collision on walls

        if self.agent1.x == agent2_old_x and self.agent1.y == agent2_old_y and \
                        self.agent2.x == agent1_old_x and self.agent2.y == agent1_old_y:
            self.agent1.x, self.agent1.y = agent1_old_x, agent1_old_y
            self.agent2.x, self.agent2.y = agent2_old_x, agent2_old_y

        if self.agent1.x == self.agent2.x and self.agent1.y == self.agent2.y:
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
                        self.agent1.load(trash)
                        break

        if agent2_action == 5:
            if self.agent2.load_status is None:
                for trash in self.trash_objects:
                    if trash.get_status() == 0 and trash.x == self.agent2.x and trash.y == self.agent2.y:
                        self.agent2.load(trash)
                        break

        # place
        if agent1_action == 6:
            if self.agent1.is_loading():
                trash = self.agent1.place()
                if self.dump1.x == self.agent1.x and self.dump1.y == self.agent1.y:
                    self.dump1.receive_trash(trash)
                    r += 0.5
                if self.dump2.x == self.agent1.x and self.dump2.y == self.agent1.y:
                    self.dump2.receive_trash(trash)
                    r += 0.5

        if agent2_action == 6:
            if self.agent2.is_loading():
                trash = self.agent2.place()
                if self.dump1.x == self.agent2.x and self.dump1.y == self.agent2.y:
                    self.dump1.receive_trash(trash)
                    r += 0.5
                if self.dump2.x == self.agent2.x and self.dump2.y == self.agent2.y:
                    self.dump2.receive_trash(trash)
                    r += 0.5

        # terminate condition
        trash_collected = self.dump1.dump + self.dump2.dump
        if len(trash_collected) == 2:
            done = True

        s_ = self.train_render()

        m1_ = 0 if self.agent1.load_status is None else 1
        m2_ = 0 if self.agent2.load_status is None else 1

        return s_, m1_, m2_, ir1, ir2, r, r, done, reached1, reached2

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
            for (x,y) in self.walls:
                a[y + 1, x + 1, i] = 0.5

        # print(a[:,:,1])
        return a

    def render_env(self):
        a = self.render_contribute_metrix()

        # b = scipy.misc.imresize(a[:, :, 0], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        # c = scipy.misc.imresize(a[:, :, 1], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')
        # d = scipy.misc.imresize(a[:, :, 2], [5 * self.size_y, 5 * self.size_x, 1], interp='nearest')

        b = scipy.misc.imresize(a[:, :, 0], [self.render_size, self.render_size, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [self.render_size, self.render_size, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [self.render_size, self.render_size, 1], interp='nearest')

        if self.render_type == 0:
            e = b * 0.299 + c * 0.587 + d * 0.114
            # e = np.array(e, dtype=np.float32).reshape([5 * self.size_y, 5 * self.size_x, 1])
            e = np.stack([e, e, e], axis=2)
        else:
            e = np.stack([b, c, d], axis=2)

        return e

    def train_render(self):
        a = self.contribute_metrix()
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

    def encodeGoal(self, goal):
        a = np.ones([self.size_y + 2, self.size_x + 2], dtype=np.float32)
        a[1:-1, 1:-1] = 0

        if goal == 0:
            a[self.dump1.y + 1, self.dump1.x + 1] = 0.5

        if goal == 1:
            a[self.dump2.y + 1, self.dump2.x + 1] = 0.5

        if goal == 2:
            if self.trash1.get_status() == 0:
                a[self.trash1.y + 1, self.trash1.x + 1] = 1

        if goal == 3:
            if self.trash2.get_status() == 0:
                a[self.trash2.y + 1, self.trash2.x + 1] = 1


        e = a[1:-1, 1:-1].flatten()

        return e.tolist()

    def checkIfReachGoal(self, x, y, g):
        reach = False
        if g == 0:
            if self.dump1.x == x and self.dump1.y == y:
                reach = True
        elif g == 1:
            if self.dump2.x == x and self.dump2.x == y:
                reach = True
        elif g == 2:
            if self.trash1.x == x and self.trash1.y == y:
                reach = True
        elif g == 3:
            if self.trash2.x == x and self.trash2.y == y:
                reach = True

        return reach