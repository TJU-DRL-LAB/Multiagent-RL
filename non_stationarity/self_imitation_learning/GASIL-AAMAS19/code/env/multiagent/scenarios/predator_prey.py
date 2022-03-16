import numpy as np
from env.multiagent.core import World, Agent, Landmark
from env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    # good_colors = [np.array([0, 0.0666, 0.275]), np.array([0.2627451, 0.41960784, 0.67843137]), np.array([0.63529412, 0.81176471, 0.99607843])]
    good_colors = [np.array([0x66, 0x00, 0x66,]) / 255, np.array([0x00, 0x99, 0xff]) / 255, np.array([0x66,0xff,0xff])/255]

    # np.array([0.63529412, 0.81176471, 0.99607843]), np.array([0.2627451 , 0.41960784, 0.67843137]), np.array([0, 0.0666, 0.275]), np.array( [0.26666667, 0.55686275, 0.89411765])  np.array([0.11764706, 0.28235294, 0.56078431]
    # good_colors = [np.array([0.35, 0.4, 0.35]), np.array([0.35, 0.7, 0.35]), np.array([0.35, 1, 0.35])]
    # a2 cf fe     44 8e e4    43 6b ad    1e 48 8f  00 11 46
    # np.array([[162, 207, 254,],[68, 142, 228],[67, 107, 173], [30, 72, 143]])

    # c0 fb 2d     58 bc 08    e5 00 00
    # np.array([[192, 251, 45],[88, 188, 8,], [229,0,0]])
    # action number
    num_good_agents = 3
    # agent number
    num_adversaries = 2

    # self action is left side
    max_step_before_punishment = 10
    # print("max_step_before_punishment: ", max_step_before_punishment)

    # 左边是我，上面是伙伴
    r_inner = [
        [11, -30, 0,   -30],
        [-30, 7,  6,   -20],
        [0,   6,  5,     0],

        [-30,-20, 0,     0]
    ]

    r_outer = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],

        [0, 0, 0, 0],
    ]

    rewards_range = [
        r_inner, r_outer
    ]
    print('reward definition: ', rewards_range)

    prey_init_pos = np.random.uniform(-1, +1, 2)

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.num_good_agents
        # good_sizes = [0.06, 0.055, 0.05]
        good_sizes = [0.05] * 3

        num_adversaries = self.num_adversaries
        # 0 ~ num_adversaries: are adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.spread_rewards = True if i < num_adversaries else False
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.size = 0.05 if agent.adversary else 0.075
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # for reward calculating...
        self.adversary_episode_max_rewards = [0] * self.num_adversaries
        self.end_without_supports = [False] * self.num_adversaries

        # print("reset world....")
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = self.good_colors[i - self.num_adversaries] if not agent.adversary else np.array(
                [0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            # print('world.dim_p: ', world.dim_p)  # 2
            # print('world.dim_c: ', world.dim_c)  # 2
            if agent.adversary:
                agent.reset_predator()
            else:
                agent.reset_prey()

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
            # print('agent state: ', agent.state)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, predator, prey, collision_level=Agent.distance_spread[1]):
        delta_pos = predator.state.p_pos - prey.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = prey.size + predator.size * collision_level
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # define the reward (coordination)
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        # doesn't change (rewarded only when there is a real collision)
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    # print('collision...')
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                # print('agent is out...')
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        # TODO: if bounded, then hidden this code.
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def return_collision_good_agent_idx(self, predator, good_agents, distance_range):
        for idx, good in enumerate(good_agents):
            if self.is_collision(predator, good, distance_range):
                return idx
        return -1

    def set_arrested(self, prey):
        prey.arrested = True

    def set_unarrested(self, prey):
        prey.arrested = False

    def set_watched(self, prey):
        prey.watched = True

    def set_unwatched(self, prey):
        prey.watched = False

    def set_pressed(self, prey):
        prey.pressed = True

    def set_unpressed(self, prey):
        prey.pressed = False

    def set_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == -1:  # 没抓过
            predator.press_prey_idx = prey_idx
            predator.press_down_step += 1
            # print("predator ", predator.idx, ": ", predator.press_down_step, ' prey: ', prey_idx)
        elif predator.press_prey_idx == prey_idx:  # 抓了同一个
            predator.press_down_step += 1
            # print("predator ", predator.idx, ": ", predator.press_down_step, ' prey: ', prey_idx)
        else:  # 没抓同一个
            predator.press_prey_idx = prey_idx
            predator.press_down_step = 1

    def release_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == prey_idx:
            predator.reset_predator()

    def set_arrested_pressed_watched(self, world):
        # print('set_arrested_pressed_watched')

        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        for dis_idx, distance_range in enumerate(Agent.distance_spread[1:]):
            for prey_idx, prey in enumerate(good_agents):
                collision_num = 0
                for predator in adversaries:
                    if self.is_collision(predator, prey, collision_level=distance_range):
                        collision_num += 1
                        # 处理 predator 状态
                        if dis_idx == 0:
                            self.set_predator_pressed(predator, prey_idx)
                        elif dis_idx == 1:
                            pass
                    else: # 没抓当前这个
                        if dis_idx == 0:
                            self.release_predator_pressed(predator, prey_idx)


                if dis_idx == 0:
                    if collision_num == 2:
                        self.set_arrested(prey)
                        # print("Setting arrested....")
                    elif collision_num == 1:
                        self.set_pressed(prey)
                        # print("Setting pressed....")
                    elif collision_num == 0:
                        self.set_unarrested(prey)
                        self.set_unpressed(prey)

                elif dis_idx == 1:
                    if collision_num >= 1:
                        self.set_watched(prey)
                    else:
                        self.set_unwatched(prey)




    # define the reward (coordination) for agent
    def adversary_reward(self, agent, world):
        # step_penalize = -0.1
        step_penalize = 0
        # Adversaries are rewarded for collisions with agents
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent.collide:
            # TODO: For each action
            # 有近到远，对于每一个猎物，看看有没有同时抓一个，
            # 否则看看由近到远看看有没有分别去抓不同的
            for dis_idx, distance_range in enumerate(agent.distance_spread[1:]):
                for prey_idx, prey in enumerate(good_agents):
                    collision_num = 0
                    self_collision = 0
                    for predator in adversaries:
                        if self.is_collision(predator, prey, collision_level=distance_range):
                            collision_num += 1
                            if predator == agent:  # 自己
                                self_collision += 1
                    if collision_num == 2 and self_collision == 1:  # 由远到近已经合作去抓了一个
                        # reward 增长
                        rew = self.rewards_range[dis_idx][prey_idx][prey_idx]
                        rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                        self.adversary_episode_max_rewards[agent.idx] = rew
                        return rt_rew

                    elif collision_num == 1:
                        if self_collision == 1:  # 我抓了当前这个，队友没抓
                            # TODO: 所有队友(当前队友只有一个)
                            partners = [a for a in adversaries if a != agent]
                            partner_collision_action_idx = self.return_collision_good_agent_idx(partners[0], good_agents,distance_range)
                            if partner_collision_action_idx != -1:  # 在当前观察级别下，队友确实在抓另一个（没合作）
                                rew = self.rewards_range[dis_idx][prey_idx][partner_collision_action_idx]
                                if rew > 0:  # 次优结
                                    rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                    self.adversary_episode_max_rewards[agent.idx] = rew
                                    # print("合作了，次优结: ", rew)
                                    return rt_rew
                                else:
                                    # print("没合作,是惩罚: ", rew)
                                    return rew
                            else:  # 队友没在抓其他的，但是也没合作
                                if dis_idx == 0 and agent.press_down_step > self.max_step_before_punishment:
                                    # print('agent.press_down_step > self.max_step_before_punishment', agent.press_down_step, '; prey: ', prey_idx)
                                    # TODO 队友没有在几步之内救自己，受到惩罚而结束
                                    self.end_without_supports[agent.idx-0] = True
                                    # 惩罚 reward
                                    rew = self.rewards_range[dis_idx][prey_idx][3]
                                    # TODO: 要加生命值的话，在这儿加
                                    return rew
                                else: # TODO 这里考虑还要不要惩罚（按着的时候）
                                    # return 0
                                    # print("max_step_before_punishment: ", self.max_step_before_punishment)
                                    pass

                        elif self_collision == 0: # 我没抓当前这个，队友抓了
                            self_collision_action_idx = self.return_collision_good_agent_idx(agent,good_agents,distance_range)
                            if self_collision_action_idx != -1:  # 在当前观察级别下，我确实在抓另一个（没与队友合作）
                                rew = self.rewards_range[dis_idx][self_collision_action_idx][prey_idx]
                                if rew > 0:  # 次优结
                                    rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                    self.adversary_episode_max_rewards[agent.idx] = rew
                                    # print("合作了，次优结: ", rew)
                                    return rt_rew
                                else:
                                    # print("没合作,是惩罚: ", rew)
                                    return rew
                            else: # 我没抓另一个，但也没与队友合作
                                partners = [a for a in adversaries if a != agent]
                                if dis_idx == 0 and partners[0].press_down_step > self.max_step_before_punishment:
                                    # print('partners[0].press_down_step > self.max_step_before_punishment', partners[0].press_down_step, '; prey: ', partners[0].press_prey_idx)
                                    # TODO 我没有在几步之内去救队友 (受到惩罚而结束)
                                    self.end_without_supports[agent.idx-0] = True
                                    # 惩罚 reward
                                    rew = self.rewards_range[dis_idx][3][prey_idx]
                                    # TODO: 要加生命值的话，在这儿加
                                    return rew
                                else: # TODO 这里考虑还要不要惩罚（按着的时候）
                                    # return 0
                                    pass
        return step_penalize

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue  # 跳过当前 agent
            comm.append(other.state.c)  # 2
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 2
            if not other.adversary:  # 我的观察里有被捕食者，添加其速度
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        # 2, 2, 2 * 2, 3 * 2, [2] = 16 / 14

    # Add the end condition.
    def done(self, agent, world):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # TODO: For each action
        collision_adv = 0
        collision_agents = set()
        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            for adv in adversaries:
                if self.is_collision(adv, ag):
                    collision_adv += 1
                    collision_agents.add(adv.idx)
        # For coordination
        if len(collision_agents) == self.num_adversaries or any(self.end_without_supports):
            # for ag in adversaries:
            #     ag.print_info()
            # for ag in agents:
            #     ag.print_info()
            return True
        return False

    def collision_number(self, agent, world):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        result = {idx: 0 for idx in range(self.num_good_agents)}
        # TODO: For each action
        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            collision_adv = 0
            for adv in adversaries:
                if self.is_collision(ag, adv):
                    collision_adv += 1
            # For coordination
            if collision_adv == 2:
                result[idx] = 1
        return result
