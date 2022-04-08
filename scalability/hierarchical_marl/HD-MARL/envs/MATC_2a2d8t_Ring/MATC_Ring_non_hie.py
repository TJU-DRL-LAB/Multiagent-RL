#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc


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
    def __init__(self, width=11, height=11, agent_num=2, trash_num=2, render_size=84, render_type=1, is_fixed=True):
        self.size_x = width
        self.size_y = height
        self.objects = []

        self.agent_num = agent_num
        self.trash_num = trash_num

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: stay, 5: return, 6: load, 7: place,
        self.action_num = 7

        # 0: dump 1, 1: dump 2, 2: trash 1, 3: trash 2, 4: trash 3, 5: load, 6: place
        self.goal_num = 12
        self.render_size = render_size
        # 0 for grey, 1 for rgb
        self.render_type = render_type

        self.is_fixed = is_fixed
        # self.done = False
        self.reset()

    def reset(self):
        self.dump1 = DumpObj(coordinates=(4, 6), mark=1)
        self.dump2 = DumpObj(coordinates=(6, 4), mark=2)

        self.agent1 = AgentObj(coordinates=(4, 4), type=2, mark=1, name='agent1')
        self.agent2 = AgentObj(coordinates=(6, 6), type=0, mark=2, name='agent2')

        # TODO 11x11, CENTER POINT (5,5), (4,1), (6,1), (9,4), (9,6), (6,9), (4,9), (1,6), (1,4)
        self.trash_objects = []
        self.trash_pos = [(4,1), (6,1), (9,4), (9,6), (6,9), (4,9), (1,6), (1,4)]
        self.trash1 = TrashObj(coordinates=(self.trash_pos[0]), mark=1)
        self.trash2 = TrashObj(coordinates=(self.trash_pos[1]), mark=2)
        self.trash3 = TrashObj(coordinates=(self.trash_pos[2]), mark=3)
        self.trash4 = TrashObj(coordinates=(self.trash_pos[3]), mark=4)
        self.trash5 = TrashObj(coordinates=(self.trash_pos[4]), mark=5)
        self.trash6 = TrashObj(coordinates=(self.trash_pos[5]), mark=6)
        self.trash7 = TrashObj(coordinates=(self.trash_pos[6]), mark=7)
        self.trash8 = TrashObj(coordinates=(self.trash_pos[7]), mark=8)

        self.trash_objects.append(self.trash1)
        self.trash_objects.append(self.trash2)
        self.trash_objects.append(self.trash3)
        self.trash_objects.append(self.trash4)
        self.trash_objects.append(self.trash5)
        self.trash_objects.append(self.trash6)
        self.trash_objects.append(self.trash7)
        self.trash_objects.append(self.trash8)

        s = self.train_render()
        return s


    def step(self, agent1_action, agent2_action, nom=False):
        assert agent1_action in range(self.action_num), 'agent1 takes wrong action'
        assert agent2_action in range(self.action_num), 'agent2 takes wrong action'

        agent1_old_x, agent1_old_y = self.agent1.x, self.agent1.y
        agent2_old_x, agent2_old_y = self.agent2.x, self.agent2.y

        r1 = 0.0
        r2 = 0.0
        done = False

        # move-actions
        if agent1_action in range(5):
            agent1_action_return = self.agent1.move(agent1_action, env_x_size=self.size_x, env_y_size=self.size_y)
        if agent2_action in range(5):
            agent2_action_return = self.agent2.move(agent2_action, env_x_size=self.size_x, env_y_size=self.size_y)

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
                    r1 = 0.5
                if self.dump2.x == self.agent1.x and self.dump2.y == self.agent1.y:
                    self.dump2.receive_trash(trash)
                    r1 = 0.5

        if agent2_action == 6:
            if self.agent2.is_loading():
                trash = self.agent2.place()
                if self.dump1.x == self.agent2.x and self.dump1.y == self.agent2.y:
                    self.dump1.receive_trash(trash)
                    r2 = 0.5
                if self.dump2.x == self.agent2.x and self.dump2.y == self.agent2.y:
                    self.dump2.receive_trash(trash)
                    r2 = 0.5

        # terminate condition
        trash_collected = self.dump1.dump + self.dump2.dump
        if len(trash_collected) == 8:
            done = True

        s_ = self.train_render()

        m1_ = 0 if self.agent1.load_status is None else 1
        m2_ = 0 if self.agent2.load_status is None else 1

        return s_, m1_, m2_, r1, r2, done

    def contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3], dtype=np.float32)
        a[1:-1, 1:-1, :] = 0

        a[self.dump1.y + 1, self.dump1.x + 1, 2] = 1
        a[self.dump2.y + 1, self.dump2.x + 1, 2] = 1

        for trash in self.trash_objects:
            if trash.get_status() == 0:
                a[trash.y + 1, trash.x + 1, 1] = 1

        a[self.agent1.y + 1, self.agent1.x + 1, 0] = 1
        a[self.agent2.y + 1, self.agent2.x + 1, 0] = 1

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
            a[self.agent1.y + 1, self.agent1.x + 1, i] = 1 if i == self.agent1.type else 0
            a[self.agent2.y + 1, self.agent2.x + 1, i] = 1 if i == self.agent2.type else 0

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

        # TODO dump tag 0.5 -> 1.0
        if goal == 0:
            a[self.dump1.y + 1, self.dump1.x + 1] = 1

        if goal == 1:
            a[self.dump2.y + 1, self.dump2.x + 1] = 1

        if goal == 2:
            if self.trash1.get_status() == 0:
                a[self.trash1.y + 1, self.trash1.x + 1] = 1

        if goal == 3:
            if self.trash2.get_status() == 0:
                a[self.trash2.y + 1, self.trash2.x + 1] = 1

        if goal == 4:
            if self.trash3.get_status() == 0:
                a[self.trash3.y + 1, self.trash3.x + 1] = 1

        if goal == 5:
            if self.trash4.get_status() == 0:
                a[self.trash4.y + 1, self.trash4.x + 1] = 1

        if goal == 6:
            if self.trash5.get_status() == 0:
                a[self.trash5.y + 1, self.trash5.x + 1] = 1

        if goal == 7:
            if self.trash6.get_status() == 0:
                a[self.trash6.y + 1, self.trash6.x + 1] = 1

        if goal == 8:
            if self.trash7.get_status() == 0:
                a[self.trash7.y + 1, self.trash7.x + 1] = 1

        if goal == 9:
            if self.trash8.get_status() == 0:
                a[self.trash8.y + 1, self.trash8.x + 1] = 1

        e = a[1:-1, 1:-1].flatten()

        return e.tolist()

    def checkIfReachGoal(self, x, y, g):
        if g == 0:
            if self.dump1.x == x and self.dump1.y == y:
                return True
        elif g == 1:
            if self.dump2.x == x and self.dump2.y == y:
                return True
        elif g == 2:
            if self.trash1.x == x and self.trash1.y == y:
                return True
        elif g == 3:
            if self.trash2.x == x and self.trash2.y == y:
                return True
        elif g == 4:
            if self.trash3.x == x and self.trash3.y == y:
                return True
        elif g == 5:
            if self.trash4.x == x and self.trash4.y == y:
                return True
        elif g == 6:
            if self.trash5.x == x and self.trash5.y == y:
                return True
        elif g == 7:
            if self.trash6.x == x and self.trash6.y == y:
                return True
        elif g == 8:
            if self.trash7.x == x and self.trash7.y == y:
                return True
        elif g == 9:
            if self.trash8.x == x and self.trash8.y == y:
                return True
        return False