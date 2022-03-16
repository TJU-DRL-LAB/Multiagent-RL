import pickle


# def discount_with_dones(rewards, dones, gamma):
#     discounted = []
#     r = 0
#     for reward, done in zip(rewards[::-1], dones[::-1]):
#         r = reward + gamma*r*(1.-done) # fixed off by one bug
#         discounted.append(r)
#     return discounted[::-1]

def discount_with_dones(trajectory, gamma):
    '''
    :param trajectory: (obs_t, action, reward, obs_tp1, done)
    :param gamma:
    :return:
    '''
    discounted = []
    r = 0
    # positive_return = 0
    normal_return = 0
    for item in trajectory[::-1]:
        r = item[2] + gamma * r * (1. - item[-1])  # fixed off by one bug
        # positive_return = (item[2] if item[2] >0 else 0) + positive_return * (1. - item[-1])
        normal_return = item[2] + normal_return * (1. - item[-1])
        discounted.append(r)
    return discounted[::-1], normal_return


def add_episode(buffer, trajectory, gamma):
    returns, Return = discount_with_dones(trajectory, gamma)
    # print(returns)
    end = len(returns) - 1
    start = 0
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        buffer.add(obs_t, action, reward, obs_tp1, float(done), float(end - start), R)
        start += 1
    buffer.mean_returns.append(buffer.current_mean_return)

def add_positive_episode(buffer, trajectory, gamma, mean=0):
    returns, Return = discount_with_dones(trajectory, gamma)
    # 按照 discount 是 1 计算
    # original_returns = discount_with_dones(trajectory, 1)
    if returns[-1] >= mean:
        end = len(returns) - 1
        start = 0
        for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
            buffer.add(obs_t, action, reward, obs_tp1, float(done), float(end-start), R)
            start += 1
        buffer.mean_returns.append(buffer.current_mean_return)

def add_trajectory(buffer, trajectory, gamma):
    returns, Return = discount_with_dones(trajectory, gamma)
    # 按照 discount 是 1 计算
    # original_returns = discount_with_dones(trajectory, 1)
    end = len(returns) - 1
    start = 0
    new_trajectory = []
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        new_trajectory.append([obs_t, action, reward, obs_tp1, float(done), float(end - start), R])
        start += 1
    # TODO: 只计算最后state的return
    # buffer.add(new_trajectory, returns[-1])
    # TODO: 计算所有state的return
    buffer.add(new_trajectory, Return)


def add_positive_trajectory(buffer, trajectory, gamma, mean=0):
    returns, Return = discount_with_dones(trajectory, gamma)
    # 按照 discount 是 1 计算
    # original_returns = discount_with_dones(trajectory, 1)
    end = len(returns) - 1
    start = 0
    new_trajectory = []
    for (obs_t, action, reward, obs_tp1, done), R in zip(trajectory, returns):
        new_trajectory.append([obs_t, action, reward, obs_tp1, float(done), float(end - start), R])
        start += 1
    if returns[-1] >= mean:
        # print("Positive occur: ", returns[-1], ', Mean: ', mean, 'length: ', len(returns), len(trajectory))
        # print(returns)
        # TODO: 只计算最后state的return
        buffer.add(new_trajectory, returns[-1])
        # TODO: 计算所有state的return
        # buffer.add(new_trajectory, Return)


def reload_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data
