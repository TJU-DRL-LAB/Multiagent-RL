"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.RLModels import \
    HistoryEncoder, HistoryIDEncoder, CurrentEncoder, HistoryIdentity, \
    HistoryModel, CurrentModel, \
    MixedPolicy, SinglePolicy
from onmt.Utils import use_gpu

from cocoa.io.utils import read_pickle
from neural import make_model_mappings


def make_embeddings(opt, word_dict, emb_length, for_encoder=True):
    return nn.Embedding(len(word_dict), emb_length)


def make_identity(opt, intent_size, hidden_size, hidden_depth, identity_dim=2, emb=None):
    diaact_size = (intent_size+1+1)
    extra_size = 3
    if hidden_size is None:
        hidden_size = opt.hidden_size
    identity = HistoryIdentity(diaact_size * 2, hidden_size, extra_size,
                               identity_dim=identity_dim, hidden_depth=hidden_depth,
                               uttr_emb=emb)

    return identity

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
    if use_history:
        extra_size = 3
        # + pmask
        diaact_size += 1
        if identity is None:
            encoder = HistoryIDEncoder(None, diaact_size * 2, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth, rnn_state=True)
        else:
            # encoder = HistoryIDEncoder(identity, diaact_size*2+extra_size, embeddings, output_size,
            #                            hidden_depth=hidden_depth)
            encoder = HistoryIDEncoder(identity, diaact_size * 2, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth, rnn_state=True)
    else:
        if identity is None:
            encoder = CurrentEncoder(diaact_size*opt.state_length+extra_size, embeddings, output_size,
                                     hidden_depth=hidden_depth)
        else:
            extra_size = 3
            # + pmask
            diaact_size += 1
            encoder = HistoryIDEncoder(identity, diaact_size * opt.state_length, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth)

    return encoder


def make_decoder(opt, encoder_size, intent_size, hidden_size, price_action=False, output_value=False, hidden_depth=2):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if output_value:
        return SinglePolicy(encoder_size, intent_size, hidden_size=hidden_size, hidden_depth=hidden_depth)
    if price_action:
        return MixedPolicy(encoder_size, intent_size, 4, hidden_size=hidden_size, hidden_depth=hidden_depth)
    return MixedPolicy(encoder_size, intent_size, 1, hidden_size=hidden_size, hidden_depth=hidden_depth)
    # return PolicyDecoder(encoder_size=encoder_size, intent_size=intent_size)


def load_test_model(model_path, opt, dummy_opt, new_opt, model_type='sl', load_type=None, exclude={}):
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

    if model_type == 'sl':
        model = make_sl_model(model_opt, mappings, use_gpu(opt), checkpoint)
        model.eval()

        return mappings, model, model_opt
    else:
        actor, critic, tom = make_rl_model(model_opt, mappings, use_gpu(opt), checkpoint, load_type, exclude=exclude)
        actor.eval()
        critic.eval()
        tom.eval()
        return mappings, (actor, critic, tom), model_opt


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
        model.load_state_dict(checkpoint[model_name])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                # don't init embedding
                if p.requires_grad:
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)

def make_rl_model(model_opt, mappings, gpu, checkpoint=None, load_type='from_sl', exclude={}):
    # print('rnn-type', model_opt.rnn_type)
    # print('h-size', model_opt.hidden_depth)
    # print('th-d', model_opt.tom_hidden_size)
    intent_size = mappings['lf_vocab'].size

    # Make encoder.
    # For A&C fix encoder
    # For ToM using new encoder
    # Fix embedding
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    rl_encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    # tom_encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size, use_history=True)
    if load_type is not None and (model_opt.tom_model in ['history', 'naive'] or len(exclude) == 0):
    # if model_opt.tom_model in ['history', 'naive']:

        use_history= model_opt.tom_model == "history" or (load_type is not None and len(exclude) == 0)
        tom_encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.tom_hidden_size,
                                   use_history=use_history, hidden_depth=model_opt.tom_hidden_depth,
                                   identity=None, hidden_size=model_opt.tom_hidden_size)
    else:
        id_emb, tom_emb = None, src_embeddings
        if model_opt.tom_model in ['uttr_id_history_tom', 'uttr_fid_history_tom', 'uttr_id']:
            id_emb, tom_emb = src_embeddings, None

        tom_identity = make_identity(model_opt, intent_size, model_opt.id_hidden_size,
                                     hidden_depth=model_opt.id_hidden_depth, identity_dim=7, emb=id_emb)
        tom_encoder = make_encoder(model_opt, tom_emb, intent_size, model_opt.tom_hidden_size,
                                   use_history=('history' in model_opt.tom_model), hidden_depth=model_opt.tom_hidden_depth,
                                   identity=tom_identity, hidden_size=model_opt.tom_hidden_size)
        if model_opt.tom_model == 'uttr_fid_history_tom':
            tom_encoder.fix_identity = False

    rl_encoder.fix_emb = True
    tom_encoder.fix_emb = True
    if tom_encoder.identity is not None:
        tom_encoder.identity.fix_emb = True
    src_embeddings.weight.requires_grad_(False)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']
    actor_decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size, model_opt.hidden_size, price_action=True)
    critic_decoder = make_decoder(model_opt, model_opt.hidden_size, 1, model_opt.hidden_size, output_value=True)
    tom_decoder = make_decoder(model_opt, model_opt.tom_hidden_size, intent_size, model_opt.tom_hidden_size, hidden_depth=1)

    # tom_decoder = make_decoder(model_opt, model_opt.hidden_size, 2, model_opt.hidden_size, output_value=True)
    # print('decoder', decoder)

    actor_model = CurrentModel(rl_encoder, actor_decoder, fix_encoder=True)
    critic_model = CurrentModel(rl_encoder, critic_decoder, fix_encoder=True)

    tom_model = HistoryModel(tom_encoder, tom_decoder)

    #TODO: use for tom_identity test
    # tom_model = make_encoder(model_opt, src_embeddings, intent_size, 2, use_history=True)

    print('load type:', load_type)
    if load_type == 'from_sl' and checkpoint.get('tom') is not None:
        print('In fact, load from rl!')
        load_type = 'from_rl'

    # TODO: Random init, load from different tom
    random_init_rl = False
    # First
    if load_type == 'from_sl':
        if random_init_rl:
            transfer_critic_model(actor_model, checkpoint, model_opt, 'model')
        else:
            transfer_actor_model(actor_model, checkpoint, model_opt, 'model')
        transfer_critic_model(critic_model, checkpoint, model_opt, 'model')
        init_model(tom_model, None, model_opt, 'tom')
    else:
        init_model(actor_model, checkpoint, model_opt, 'model')
        init_model(critic_model, checkpoint, model_opt, 'critic')
        init_model(tom_model, None if exclude.get('tom') else checkpoint, model_opt, 'tom')

    if gpu:
        actor_model.cuda()
        critic_model.cuda()
        tom_model.cuda()
    else:
        actor_model.cpu()
        critic_model.cpu()
        tom_model.cpu()

    return actor_model, critic_model, tom_model

def make_sl_model(model_opt, mappings, gpu, checkpoint=None):
    intent_size = mappings['lf_vocab'].size

    # Make encoder.
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    # print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']

    decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size, model_opt.hidden_size)
    # print('decoder', decoder)

    model = CurrentModel(encoder, decoder)

    init_model(model, checkpoint, model_opt)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
