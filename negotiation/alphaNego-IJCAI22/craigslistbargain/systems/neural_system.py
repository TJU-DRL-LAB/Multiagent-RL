import os
import argparse
from collections import namedtuple
from onmt.Utils import use_gpu

from cocoa.systems.system import System
from cocoa.sessions.timed_session import TimedSessionWrapper
from cocoa.core.util import read_pickle, read_json
from cocoa.neural.beam import Scorer

from neural.generator import get_generator
from sessions.neural_session import PytorchNeuralSession
from sessions.tom_session import PytorchNeuralTomSession
from neural import rl_model_builder, get_data_generator, make_model_mappings
from neural.preprocess import markers, TextIntMap, Preprocessor, Dialogue
from neural.batcher_rl import DialogueBatcherFactory
from neural.utterance import UtteranceBuilder
from neural.nlg import IRNLG
from neural.batcher_rl import RawBatch
import sys
import re
import options


def get_new_args(args):
    new_args = {}
    news = []
    for s in sys.argv:
        if len(s) > 2 and s[:2] == '--':
            news.append(s[2:].replace('-', '_'))
    return news


class PytorchNeuralSystem(System):
    """
    NeuralSystem loads a neural model from disk and provides a function instantiate a new dialogue agent (NeuralSession
    object) that makes use of this underlying model to send and receive messages in a dialogue.
    """
    def __init__(self, args, schema, price_tracker, model_path, timed, name=None, id=0):
        super(PytorchNeuralSystem, self).__init__()
        self.args = args
        self.schema = schema
        self.price_tracker = price_tracker
        self.timed_session = timed

        model_type = 'sl' if (name == 'pt-neural-r') or (name == 'pt-neural-s') else 'rl'

        # TODO: do we need the dummy parser?
        dummy_parser = argparse.ArgumentParser(description='duh')
        options.add_model_arguments(dummy_parser)
        options.add_data_generator_arguments(dummy_parser)
        dummy_args = dummy_parser.parse_known_args()[0]
        dummy_args_dict = dummy_args.__dict__
        new_args = get_new_args(dummy_args_dict)

        if model_type == 'sl':
            mappings, model, model_args = rl_model_builder.load_test_model(
                model_path, args, dummy_args_dict, new_args, model_type=model_type)
            actor = model
            critic = None
            tom = None
            print('sl model:', 'actor')
        else:
            # Load the model.
            exclude = {}
            print("name",name)
            if name == "tom" and hasattr(args, 'load_identity_from') and args.load_identity_from is not None:
                exclude['tom'] = True

            mappings, model, model_args = rl_model_builder.load_test_model(
                model_path, args, dummy_args_dict, new_args, model_type=model_type, load_type='from_sl', exclude=exclude)

            actor, critic, tom = model
            # Load critic from other model.
            # if name == 'tom':
            if hasattr(args, 'load_critic_from') and args.load_critic_from is not None:
                critic_path = args.load_critic_from
                _, t_model, _ = rl_model_builder.load_test_model(
                    critic_path, args, dummy_args_dict, [], model_type=model_type)
                critic = model[1]

            if hasattr(args, 'load_identity_from') and args.load_identity_from is not None:
                print('[Info] load identity from {}.'.format(args.load_identity_from))
                identity_path = args.load_identity_from
                _, t_model, _ = rl_model_builder.load_test_model(
                    identity_path, args, dummy_args_dict, [], model_type=model_type)
                # tom.encoder.identity.load_state_dict(t_model[2].encoder.identity.state_dict())
                tom = t_model[2]

            if hasattr(args, 'ban_identity') and args.ban_identity:
                print('[Info] Identity banned.')
                tom.encoder.ban_identity = True

            # print('rl models', actor)
            # print(critic)
            # print(tom)
            print('rl models', 'actor')
            print('critic')
            print('tom')

        self.model_name = model_args.model
        vocab = mappings['utterance_vocab']
        lf_vocab = mappings['lf_vocab']
        # print(vocab.word_to_ind)
        self.mappings = mappings

        from neural.generator import LFSampler
        generator = LFSampler(actor, lf_vocab, 1, max_length=args.max_length, cuda=use_gpu(args), model_type=model_type)
        if name != 'sl':
            tom_generator = LFSampler(tom, lf_vocab, 1, max_length=args.max_length, cuda=use_gpu(args), model_type='tom')
        else:
            tom_generator = None

        # Price Tracker
        builder = UtteranceBuilder(vocab, args.n_best, has_tgt=True)
        
        nlg_module = IRNLG(args)

        preprocessor = Preprocessor(schema, price_tracker, model_args.entity_encoding_form,
                model_args.entity_decoding_form, model_args.entity_target_form)
        textint_map = TextIntMap(vocab, preprocessor)
        lfint_map = TextIntMap(lf_vocab, preprocessor)
        remove_symbols = map(vocab.to_ind, (markers.EOS, markers.PAD))
        use_cuda = use_gpu(args)

        kb_padding = mappings['kb_vocab'].to_ind(markers.PAD)
        # print('args: ', model_args.dia_num, model_args.state_length)
        dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model=self.model_name,
            kb_pad=kb_padding,
            mappings=mappings, num_context=model_args.num_context,
            dia_num=model_args.dia_num, state_length=model_args.state_length)

        # TODO: class variable is not a good way to do this
        Dialogue.preprocessor = preprocessor
        Dialogue.textint_map = textint_map
        Dialogue.lfint_map = lfint_map
        Dialogue.mappings = mappings
        Dialogue.num_context = model_args.num_context

        RawBatch.init_vocab(lf_vocab)


        Env = namedtuple('Env', ['model', 'vocab', 'preprocessor', 'textint_map',
            'stop_symbol', 'remove_symbols', 'gt_prefix', 'lfint_map',
            'max_len', 'dialogue_batcher', 'cuda', 'lf_vocab',
            'dialogue_generator', 'utterance_builder', 'model_args', 'critic', 'usetom', 
            'name', 'price_strategy', 'tom_type', 'nlg_module', 'tom_generator', 'tom_model', 'id', 'model_type', 'args'])
        self.env = Env(actor, vocab, preprocessor, textint_map,
            stop_symbol=vocab.to_ind(markers.EOS), remove_symbols=remove_symbols,
            gt_prefix=1, lfint_map=lfint_map,
            max_len=20, dialogue_batcher=dialogue_batcher, cuda=use_cuda, lf_vocab=lf_vocab,
            dialogue_generator=generator, utterance_builder=builder, model_args=model_args,
            critic=critic, usetom=(name == 'tom'), name=name,
            price_strategy=args.price_strategy, tom_type=args.tom_type, nlg_module=nlg_module,
            tom_generator=tom_generator, tom_model=tom, id=id, model_type=model_type, args=args)
        # print('usetom?:', (name == 'tom'))

    @classmethod
    def name(cls):
        return 'pt-neural'

    def new_session(self, agent, kb):
        if self.model_name in ('seq2seq', 'lf2lf'):
            if self.env.usetom:
                tom_sess = PytorchNeuralTomSession(1-agent, kb, self.env, True)
                session = PytorchNeuralSession(agent, kb, self.env, tom_sess)
            else:
                session = PytorchNeuralSession(agent, kb, self.env, False)
        else:
            raise ValueError('Unknown model name {}'.format(self.model_name))
        if self.timed_session:
            session = TimedSessionWrapper(session)
        return session
