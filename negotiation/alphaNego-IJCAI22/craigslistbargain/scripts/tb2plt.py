# Example:
# python tb2plt.py --dir p_logs/ \
# --agent0 rl_decay.1 rl_dacay.2 rl_dacay.3 rl_decay.4 rl_decay.5 \
# --agent1 tom_decay.1.b tom_decay.2.b tom_decay.3.b tom_decay.4.b tom_decay.6.b \
# --show --max-epoch 90 --agent1-move 50 --agent1-leftclip 50 --draw-type rl


# python tb2plt.py --dir apex_logs/ \
# --agent0 a2c_decay3 a2c_decay4 rl_decay.1 rl_dacay.2 rl_dacay.3 \
# --agent1 tom2 tom3 tom_decay4 tom_decay5 tom_decay6\
#               --show --max-epoch 50
# python tb2plt.py --dir apex_logs/ \
# --agent0 rl_decay.5 rl_dacay.2 rl_dacay.3 rl_decay.1 rl_decay.4 \
# --agent1 tom2 tom3 tom_decay4 tom_decay5 tom_decay6\
#               --show --max-epoch 90


# Load scalar from tensorboard logs and render with matplotlib
#   * extract_rewards_from_example(file_name):
#       Get rewards from specific examples file
#   * load_from_tensorboard
#       Get logs (include rewards) from tensorboard

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re

edge = 0.17
left_e = 0.22
up_e = 0.92
right_e = 0.98

def extract_rewards_from_example(file_name='result.txt'):
    r = re.compile('([-+]?\d+\.\d+)')
    with open(file_name,'r') as f:
        data = f.readlines()
    rewards = [[], []]
    for d in data:
        if d.find('reward: [0]') != -1:
            rewards[0].append(float(r.findall(d)[0]))
        if d.find('reward: [1]') != -1:
            rewards[1].append(float(r.findall(d)[0]))
    return [np.mean(rewards[0]), np.mean(rewards[1])]
    # print('rewards: {}, {}'.format(np.mean(rewards[0]), np.mean(rewards[1])))

def get_args():
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dir', default='logs/', help='directory of the tensorboard log file')
    parser.add_argument('--agent0', nargs='+', type=str, help='directory of the tensorboard log file')
    parser.add_argument('--agent1', nargs='+', type=str, help='directory of the tensorboard log file')
    parser.add_argument('--draw-type', choices=['sl', 'rl'], type=str, default='sl', help='plot type')
    parser.add_argument('--show', action='store_true', help='show pictures directly')
    parser.add_argument('--fig-dirs', default='./', help='direcotry of these figures')
    parser.add_argument('--font-size', type=int, default=24, help='size of fonts')
    parser.add_argument('--label-size', type=int, default=24, help='size of font on label')

    # parser.add_argument('--test', type=str, nargs='+', choices=['a', 'b', 'c'], help='size of font on label')

    parser.add_argument('--max-epoch', type=int, default=50, help='')
    parser.add_argument('--plot-step', type=int, default=5, help='')
    parser.add_argument('--train-step', type=int, default=100)
    parser.add_argument('--one-point', default=False, action='store_true')
    parser.add_argument('--item-name', default='agent0/reward', type=str)
    parser.add_argument('--agent1-leftclip', default=50, type=int)
    parser.add_argument('--agent1-move', default=0, type=int)

    args = parser.parse_args()
    return args

def load_from_tensorboard(dir):
    data = []

    # for root, dirs, files in os.walk(dir):
    #     for file in files:
    #         fpath = os.path.join(root, file)
    ea = event_accumulator.EventAccumulator(dir)
    ea.Reload()
    # print(ea.scalars.Keys())

    # val_psnr = ea.scalars.Items('val_psnr')
    # print(len(val_psnr))
    # print([(i.step, i.value) for i in val_psnr])
    return ea.scalars

def get_xy(it):
    x, y = [], []
    for i in it:
        x.append(i.step)
        y.append(i.value)
    return x, y

# Supervise Learning
# dev/train loss
def draw_sl_training_curve(rl_dirs, tom_dirs, args):

    plot_x = list(range(0, args.max_epoch, args.plot_step))
    plot_y_dict = [{}, {}]

    def aggregate_info(dirs, pyd, item_name='agent0/reward'):
        for d in dirs:
            scalars = load_from_tensorboard(d)
            xx, yy = get_xy(scalars.Items(item_name))
            for i, y in enumerate(yy):
                if args.one_point and xx[i] % (args.train_step* args.plot_step) != 0:
                    continue
                x = (xx[i] // args.train_step - 1) // args.plot_step * args.plot_step
                if not pyd.get(x):
                    pyd[x] = []
                pyd[x].append(y)

    # load rl data
    aggregate_info(rl_dirs, plot_y_dict[0], item_name=args.item_name)
    aggregate_info(tom_dirs, plot_y_dict[1], item_name=args.item_name)

    # for i in range(2):
    #     step = 50
    #     print('[{agent}]{step}: {mean}(+-{std})'
    #           .format(agent=i, step=step, mean=np.mean(plot_y_dict[i][step]), std = np.std(plot_y_dict[i][step])))


    # Draw text
    labels = ['rl', 'tom']
    x = []
    y = []
    plt.style.use('ggplot')

    # Draw train set
    for i in range(2):
        ymean = []
        ystd = []
        for j, x in enumerate(plot_x):
            yylist = plot_y_dict[i][x]
            mean = np.mean(yylist)
            std = np.std(yylist)
            ymean.append(mean)
            ystd.append(std)
        # plt.plot(plot_x, ymean, label=labels[i])
        plt.errorbar(plot_x, ymean, ystd, label=labels[i], alpha=0.7)

    plt.axhline(y=0.4, ls=":", c='gray')
    plt.axhline(y=0, ls=":", c='gray')

    plt.xlabel('Episode', fontsize=args.font_size)
    plt.ylabel('Reward', fontsize=args.font_size)
    plt.title('Sample Efficiency', fontsize=args.font_size)
    plt.tick_params(labelsize=args.label_size)
    plt.subplots_adjust(bottom=edge, left=left_e, top=up_e, right=right_e)
    plt.legend(fontsize=args.font_size)
    if args.show:
        plt.show()
    else:
        plt.savefig()

# def draw_it_var(max_var=2):
#     #f = open('record/bias%.2f.pkl' % max_var, 'rb')
#     f = open('record/var_%.1f.pkl'%max_var, 'rb')
#     loss_h = pickle.load(f)
#     # color = ['magenta', 'orange', 'green', 'red']
#     if max_var == 2:
#         plt.figure(figsize=(6.5,5))
#     else:
#         plt.figure(figsize=(6.3,5))
#     for i in range(4):
#         if i == 1:
#             plt.plot([])
#             plt.fill_between([],[],[])
#             continue
#         #loss_h[i] = loss_h[i][:70]
#         y = [np.mean(np.array(j)) for j in loss_h[i]]
#         x = list(range(len(y)))
#         std = [np.std(np.array(j)) for j in loss_h[i]]
#         sup = [y[j] + std[j] for j in range(len(y))]
#         inf = [y[j] - std[j] for j in range(len(y))]
#
#         sup = [y[j] + std[j] for j in range(len(y))]
#         inf = [y[j] - std[j] for j in range(len(y))]
#
#         plt.plot(x, y, label=labels[i], )
#         plt.fill_between(x, inf, sup, alpha=0.3)
#         plt.ylim((0, y[0]*0.4))
#         plt.xlim((0, len(x)-10))
#
#     plt.xlabel('Budget', fontsize=font_size)
#     plt.ylabel('Loss', fontsize=font_size)
#     #plt.title('Variance in [0.1, %.1f]' % (max_var), fontsize=font_size)
#     plt.tick_params(labelsize=labelsize)
#     plt.subplots_adjust(bottom=edge, left = left_e, top=up_e, right=right_e)
#     plt.legend(fontsize=font_size)
#     plt.show()

def draw_rl_tranning_curve(rl_dirs, tom_dirs, args):
    plot_x = [list(range(0, args.max_epoch, args.plot_step)),
              list(range(args.agent1_leftclip, args.max_epoch, args.plot_step))]
    plot_y_dict = [{}, {}]

    def aggregate_info(dirs, pyd, item_name='agent0/reward', right_move=0):
        for d in dirs:
            scalars = load_from_tensorboard(d)
            xx, yy = get_xy(scalars.Items(item_name))
            for i, y in enumerate(yy):
                if args.one_point and xx[i] % (args.train_step * args.plot_step) != 0:
                    continue
                x = (xx[i] // args.train_step - 1 + right_move) // args.plot_step * args.plot_step
                if not pyd.get(x):
                    pyd[x] = []
                pyd[x].append(y)

    # load rl data
    aggregate_info(rl_dirs, plot_y_dict[0], item_name=args.item_name)
    aggregate_info(tom_dirs, plot_y_dict[1], item_name=args.item_name, right_move=args.agent1_move)

    # for i in range(2):
    #     step = 50
    #     print('[{agent}]{step}: {mean}(+-{std})'
    #           .format(agent=i, step=step, mean=np.mean(plot_y_dict[i][step]), std = np.std(plot_y_dict[i][step])))

    # Draw text
    labels = ['rl', 'tom']
    x = []
    y = []
    plt.style.use('ggplot')

    # Draw train set
    for i in range(2):
        ymean = []
        ystd = []
        for j, x in enumerate(plot_x[i]):
            yylist = plot_y_dict[i][x]
            mean = np.mean(yylist)
            std = np.std(yylist)
            ymean.append(mean)
            ystd.append(std)
        # plt.plot(plot_x, ymean, label=labels[i])
        plt.errorbar(plot_x[i], ymean, ystd, label=labels[i], alpha=0.7)

    plt.axhline(y=0.4, ls=":", c='gray')
    plt.axhline(y=0, ls=":", c='gray')

    plt.xlabel('Episode', fontsize=args.font_size)
    plt.ylabel('Reward', fontsize=args.font_size)
    plt.title('Inference Fine Tuning', fontsize=args.font_size)
    plt.tick_params(labelsize=args.label_size)
    plt.subplots_adjust(bottom=edge, left=left_e, top=up_e, right=right_e)
    plt.legend(fontsize=args.font_size)
    if args.show:
        plt.show()
    else:
        plt.savefig()

if __name__ == '__main__':
    args = get_args()
    # data = load_from_tensorboard(args.dir)

    dirs0 = [args.dir+i for i in args.agent0]
    dirs1 = [args.dir+i for i in args.agent1]

    if args.draw_type == 'sl':
        # Draw sl curve
        draw_sl_training_curve(dirs0, dirs1, args)
    elif args.draw_type == 'rl':
        # Draw rl curve
        draw_rl_tranning_curve(dirs0, dirs1, args)
    elif args.draw_type == 'tom':
        pass


