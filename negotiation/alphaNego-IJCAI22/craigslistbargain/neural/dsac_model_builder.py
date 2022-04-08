"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
import copy
import onmt
import onmt.io
import onmt.Models
import onmt.modules

import numpy as np
from onmt.RLModels import CurrentEncoder, \
    HistoryModel, CurrentModel, \
    MixedPolicy, SinglePolicy, QuantileMlp
from onmt.Utils import use_gpu
from cocoa.io.utils import read_pickle
from neural import make_model_mappings
from .dsac_utils import *

class DSAC:
    def __init__(self, model_opt, mappings, gpu):
        self.intent_size = mappings['lf_vocab'].size
        self.price_size = 4
        self.device = "cuda" if gpu else "cpu"
        src_dict = mappings['utterance_vocab']
        src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
        src_embeddings.weight.requires_grad_(False)
        self.rl_encoder = make_encoder(model_opt, src_embeddings, self.intent_size, model_opt.hidden_size)
        self.rl_encoder.fix_emb = True

        self.actor_decoder = make_decoder(model_opt.hidden_size, intent_size=self.intent_size, price_size=self.price_size,
                                          hidden_size=model_opt.hidden_size, is_actor=True)
        self.critic_decoder1 = make_decoder(model_opt.hidden_size, intent_size=self.intent_size, price_size=self.price_size,
                                                   hidden_size=model_opt.hidden_size, is_actor=False)
        self.critic_decoder2 = make_decoder(model_opt.hidden_size, intent_size=self.intent_size,
                                            price_size=self.price_size,
                                            hidden_size=model_opt.hidden_size, is_actor=False)

        self.actor_model = CurrentModel(self.rl_encoder , self.actor_decoder, fix_encoder=True).to(self.device)
        self.critic_model1 = CurrentModel(self.rl_encoder , self.critic_decoder1, fix_encoder=True).to(self.device)
        self.critic_model2 = CurrentModel(self.rl_encoder, self.critic_decoder2, fix_encoder=True).to(self.device)
        self.target_actor_model = copy.deepcopy(self.actor_model)
        self.target_critic_model1 = copy.deepcopy(self.critic_model1)
        self.target_critic_model2 = copy.deepcopy(self.critic_model2)
        self.sl_model = copy.deepcopy(self.actor_model)
        hard_update(self.target_actor_model, self.actor_model)
        self.target_actor_model.eval()
        hard_update(self.target_critic_model1, self.critic_model1)
        self.target_critic_model1.eval()
        hard_update(self.target_critic_model2, self.critic_model2)
        self.target_critic_model2.eval()






        self.update_counter = 0



    def set_eval(self):
        self.actor_model.eval()
        self.sl_model.eval()
        self.critic_model1.eval()
        self.critic_model2.eval()

    def set_train(self):
        self.actor_model.train()
        self.critic_model1.train()
        self.critic_model2.train()

def make_embeddings(opt, word_dict, emb_length, for_encoder=True):
    return nn.Embedding(len(word_dict), emb_length)


def make_encoder(opt, embeddings, intent_size, output_size, use_history=False, hidden_depth=1, identity=None,
                 hidden_size=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # encoder = StateEncoder(intent_size=intent_size, output_size=output_size,
    #                     state_length=opt.state_length, extra_size=3 if opt.dia_num>0 else 0 )

    # intent + price
    diaact_size = (intent_size+1)
    extra_size = 3 + 2
    if hidden_size is None:
        hidden_size = opt.hidden_size
    if not opt.use_utterance:
        embeddings = None

    encoder = CurrentEncoder(diaact_size*opt.state_length+extra_size, embeddings, output_size,
                                 hidden_depth=hidden_depth)

    return encoder


def make_decoder(input_size, intent_size, price_size, hidden_size, is_actor=True,  hidden_depth=2, num_quantiles=32):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if is_actor: #actor mix decoder
        return MixedPolicy(input_size, intent_size, price_size, hidden_size=hidden_size, hidden_depth=hidden_depth)

    else:  # critic decoder
        return QuantileMlp(
            input_size=input_size + intent_size + price_size,
            output_size=1,
            num_quantiles=num_quantiles,
            hidden_sizes=[hidden_size, hidden_size],
        )



def load_test_model(model_path, opt, dummy_opt, new_opt, load_type=None, dsac_load_dict=None):
    if model_path is not None:
        print('Load model from {}.'.format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)


        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if (arg in new_opt) or (arg not in model_opt):
                if model_opt.__dict__.get(arg) != dummy_opt[arg]:
                    print('update: {}:{} -> {}'.format(arg, model_opt.__dict__.get(arg), dummy_opt[arg]))
                model_opt.__dict__[arg] = dummy_opt[arg]
    else:
        print('Build model from scratch.')
        checkpoint = None
        model_opt = opt

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))
    # mappings = read_pickle('{0}/{1}/vocab.pkl'.format(model_opt.mappings, model_opt.model))
    mappings = make_model_mappings(model_opt.model, mappings)


    dsac = make_rl_model(model_opt, mappings, use_gpu(opt), checkpoint, load_type, dsac_load_dict)

    return mappings, dsac, model_opt


def select_param_from(params, names):
    selected = {}
    for k in params:
        for name in names:
            if k.find(name) == 0:
                selected[k] = params[k]
                break
    return selected


def transfer_critic_model(model, checkpoint, model_opt, model_name='model'):
    # Load encoder and init decoder.
    print('Transfer sl parameters to {}.'.format(model_name))
    model_dict = model.state_dict()
    pretrain_dict = select_param_from(checkpoint[model_name], ['encoder'])
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    if model_opt.param_init != 0.0:
        for p in model.decoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)


def transfer_actor_model(model, checkpoint, model_opt, model_name='model'):
    # Load encoder and init decoder.
    print('Transfer sl parameters to {}.'.format(model_name))
    model_dict = model.state_dict()
    pretrain_dict = select_param_from(checkpoint[model_name], ['encoder', 'decoder.common_net', 'decoder.intent_net'])
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    if model_opt.param_init != 0.0:
        for p in model.decoder.price_net.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)

def init_model(model, checkpoint, model_opt, model_name='model'):

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint)
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                # don't init embedding
                if p.requires_grad:
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)

def make_rl_model(model_opt, mappings, gpu, checkpoint=None, load_type='from_sl', dsac_load_dict=None):

    dsac = DSAC(model_opt=model_opt, mappings=mappings, gpu=gpu)
    print('load type:', load_type)

    if load_type == 'from_sl':
        transfer_actor_model(dsac.actor_model, checkpoint, model_opt, 'model')
        transfer_actor_model(dsac.sl_model, checkpoint, model_opt, 'model')
        transfer_critic_model(dsac.critic_model1, checkpoint, model_opt, 'model')
        transfer_critic_model(dsac.critic_model2, checkpoint, model_opt, 'model')

    else:
        checkpoint_actor_path = dsac_load_dict.get('actor')
        checkpoint_zf1_path = dsac_load_dict.get('zf1')
        checkpoint_zf2_path = dsac_load_dict.get('zf2')
        print("loading dsac model", checkpoint_actor_path)
        if checkpoint_actor_path is not None:
            init_model(dsac.actor_model, torch.load(checkpoint_actor_path), model_opt, 'model')
        if checkpoint_zf1_path is not None:
            init_model(dsac.critic_model1, torch.load(checkpoint_zf1_path), model_opt, 'critic1')
        if checkpoint_zf2_path is not None:
            init_model(dsac.critic_model2, torch.load(checkpoint_zf2_path), model_opt, 'critic2')
    dsac.set_eval()
    return dsac

