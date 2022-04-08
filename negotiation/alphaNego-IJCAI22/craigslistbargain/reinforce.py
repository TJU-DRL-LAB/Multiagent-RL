"""
Takes two agent (Session) implementations, generates the dialogues,
and run REINFORCE.
"""

import argparse
import random
import json
import numpy as np

from onmt.Utils import use_gpu

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
import cocoa.options

from core.scenario import Scenario
from core.controller import Controller
from systems import get_system
from neural.rl_trainer import RLTrainer
from neural import build_optim
import options

def make_loss(opt, model, tgt_vocab):
    loss = ReinforceLossCompute(model.generator, tgt_vocab)
    if use_gpu(opt):
        loss.cuda()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agents', help='What kind of agent to use. The first agent is always going to be updated and the second is fixed.', nargs='*', required=True)
    parser.add_argument('--agent-checkpoints', default=[], nargs='+', help='Directory to learned models')
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether or not to have verbose prints')

    parser.add_argument('--histogram', default=False, action='store_true', help='Whether or not to show histogram of policies')
    parser.add_argument('--valid-scenarios-path', help='Output path for the validation scenarios')

    parser.add_argument('--only-run', default=False, action='store_true', help='only sample trajectories.')

    parser.add_argument('--update-oppo', action='store_true', help='update opponent')

    parser.add_argument('--model-type', default='reinforce', choices=['reinforce', 'a2c', 'critic', 'tom'], help='choise rl algorithms')
    parser.add_argument('--load-critic-from', default=None, type=str, help='load critic model from another checkpoint')

    parser.add_argument('--name', default='rl', type=str, help='Name of this experiment.')

    parser.add_argument('--mappings', help='Path to vocab mappings')
    # Initialization
    parser.add_argument('--pretrained-wordvec', nargs='+', default=['', ''],
                       help="""If a valid path is specified, then this will load
                           pretrained word embeddings, if list contains two embeddings,
                           then the second one is for item title and description""")
    parser.add_argument('--param-init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                           with support (-param_init, param_init).
                           Use 0 to not use initialization""")
    parser.add_argument('--fix-pretrained-wordvec',
                       action='store_true',
                       help="Fix pretrained word embeddings.")


    parser.add_argument('--tom-type', choices=['expectation', 'competitive', 'cooperative'], type=str, default='expectation',
                        help='tom inference type')
    parser.add_argument('--price-strategy', choices=['high', 'low', 'decay', 'neural'], type=str,
                        default='neural',
                        help='supervise agent price strategy.')

    cocoa.options.add_scenario_arguments(parser)
    options.add_data_generator_arguments(parser)
    options.add_system_arguments(parser)
    options.add_rl_arguments(parser)
    options.add_model_arguments(parser)
    args = parser.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    schema = Schema(args.schema_path)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
    valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)

    # if len(args.agent_checkpoints) == 0
    # assert len(args.agent_checkpoints) <= len(args.agents)
    if len(args.agent_checkpoints) < len(args.agents):
        ckpt = [None]*2
    else:
        ckpt = args.agent_checkpoints

    systems = [get_system(name, args, schema, False, ckpt[i]) for i, name in enumerate(args.agents)]

    rl_agent = 0
    system = systems[rl_agent]
    model = system.env.model
    loss = None
    # optim = build_optim(args, [model, system.env.critic], None)
    optim = {'model': build_optim(args, model, None),
             'critic': build_optim(args, system.env.critic, None)}
    optim['critic']._set_rate(0.05)

    scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
    from neural.a2c_trainer import RLTrainer as A2CTrainer
    trainer = A2CTrainer(systems, scenarios, loss, optim, rl_agent,
                        reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)
    trainer.learn(args)
