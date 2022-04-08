'''
Preprocess examples in a dataset and generate data for models.
'''

import random
import re
import time
import os
import numpy as np

from cocoa.core.util import read_pickle, write_pickle, read_json
from cocoa.core.entity import Entity, CanonicalEntity, is_entity
from cocoa.model.vocab import Vocabulary

from core.price_tracker import PriceTracker, PriceScaler
from core.tokenizer import tokenize
from .batcher_rl import DialogueBatcherFactory, SLBatch
from .symbols import markers
from .vocab_builder import create_mappings
from neural import make_model_mappings

__global_map = map #keep reference to the original map
lmap = lambda func, *iterable: list(__global_map(func, *iterable)) # using "map" here will cause infinite recursion
map = lmap

category_to_marker = {
        'car': markers.C_car,
        'phone': markers.C_phone,
        'housing': markers.C_housing,
        'bike': markers.C_bike,
        'furniture': markers.C_furniture,
        'electronics': markers.C_electronics,
        }


def price_filler(x):
    return x == '<price>'


class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    _uttr_buffer = {}
    def __init__(self, vocab, preprocessor):
        self.vocab = vocab
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor

    def pred_to_input(self, preds, prices=None):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred) for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def text_to_int(self, utterance, stage=None, uid=None):
        '''
        Process entities in the utterance based on whether it is used for encoding, decoding
        or ground truth.

        self._uttr_buffer is used for accelerating tom inference.
        '''
        if uid is not None:
            ret = self._uttr_buffer.get(uid)
            if ret is not None:
                return ret
        tokens = self.preprocessor.process_utterance(utterance, stage)
        ret = [self.vocab.to_ind(token) for token in tokens]
        if uid is not None:
            self._uttr_buffer[uid] = ret
        return ret

    def int_to_text(self, inds, stage=None, prices=None):
        '''
        Inverse of text_to_int.
        '''
        toks = [self.vocab.to_word(ind) for ind in inds]
        if prices is not None:
            assert len(inds) == len(prices)
            # TODO: Why do we use CanonicalEntity here?
            toks = [CanonicalEntity(value=p, type='price') if price_filler(x) else x for x, p in zip(toks, prices)]
        return toks

class Dialogue(object):
    textint_map = None
    lfint_map = None
    ROLE = -1
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    MSG = 3

    LF = 0
    # DEC = 1
    TOKEN = 1
    PACT = 2


    LF_EMPTY = None

    def __init__(self, agent, kb, uuid, model='seq2seq', hidden_price=True, update_agree=True):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.agent = agent
        self.kb = kb
        self.model = model
        self.agent_to_role = self.get_role_mapping(agent, kb)
        # KB context
        # NOTE: context_to_int will change category, title, description to integers
        self.category_str = kb.category
        self.category = kb.category
        self.title = tokenize(re.sub(r'[^\w0-9]', ' ', kb.facts['item']['Title']))
        self.description = tokenize(re.sub(r'[^\w0-9]', ' ', ' '.join(kb.facts['item']['Description'])))
        # token_turns: tokens and entitys (output of entity linking)
        self.token_turns = []
        self.tokens = []
        # parsed logical forms
        self.lf_turns = []
        self.lfs = []

        self.price_actions = []

        self.modified = []
        self.last_prices = [[], []]

        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = [[], [], []]
        # entities: has the same structure as turns, non-entity tokens are None
        self.entities = []
        self.agents = []
        self.roles = []
        self.rid = []
        self.is_int = False  # Whether we've converted it to integers
        self.num_context = None

        self.hidden_price = hidden_price
        self.update_agree = update_agree
        self.need_output = False
        self.msgs = []

        self.states = []

    @property
    def num_turns(self):
        return len(self.lf_turns)

    @property
    def num_lfs(self):
        return len(self.lfs)

    @staticmethod
    def lf_empty():
        if Dialogue.LF_EMPTY is None:
            Dialogue.LF_EMPTY = {'intent': 'start'}
        return Dialogue.LF_EMPTY

    def join_turns(self):
        for i, utterances in enumerate(self.turns):
            self.turns[i] = [x for utterance in utterances for x in utterance]

    @staticmethod
    def get_role_mapping(agent, kb):
        my_id = agent
        my_role = kb.role

        partner_id = 1 - agent
        partner_role = 'buyer' if my_role == 'seller' else 'seller'

        return {my_id: my_role, partner_id: partner_role}

    def num_tokens(self):
        return sum([len(t) for t in self.token_turns])

    def process_lf(self, lf_raw):
        intent, price = lf_raw.get('intent'), lf_raw.get('price')
        if intent is not None:
            if isinstance(intent, str):
                if self.lfint_map is not None:
                    intent = self.lfint_map.vocab.to_ind(intent)

        if price is not None:
            price = PriceScaler.scale_price(self.kb, price)
        return {'intent': intent, 'price': price}

    # Input lf is raw lf here, {'intent': 'offer', 'price': Entity() }
    def add_utterance(self, agent, utterance, lf=None, price_act=None, uid=None):
        utterance = utterance.copy()
        # Always start from the partner agent
        if len(self.agents) == 0 and agent == self.agent:
            self._add_utterance(1 - self.agent, [], lf=self.lf_empty(), price_act={})
        # Try to process lf from utterance
        # if lf is None:
        #     lf = self.process_lf(utterance)
        if lf is []:
            print('[] case: ', utterance, lf)
            assert True

        if price_act is None and lf.get('output_data') is not None:
            price_act = lf['output_data'].get('price_act')
        lf = self.process_lf(lf)
        # print("lf {}".format(lf))
        self._add_utterance(agent, utterance, lf=lf, price_act=price_act)

    def delete_last_utterance(self):
        # Always start in the middle and delete the utterance of another agent
        self.agents.pop()
        self.roles.pop()
        self.token_turns.pop()
        self.entities.pop()
        self.lf_turns.pop()
        self.modified.pop()
        self.price_actions.pop()

        # lf to int
        l = len(self.agents)
        self.rid = self.rid[:l]
        self.lfs = self.lfs[:l]
        self.tokens = self.tokens[:l]

    @classmethod
    def scale_price(cls, kb, utterance):
        return [PriceScaler.scale_price(kb, x) if is_entity(x) else x for x in utterance]

    @classmethod
    def original_price(cls, kb, utterance):
        s = [PriceScaler.unscale_price(kb, x) if is_entity(x) else x for x in utterance]
        return s

    def lf_to_tokens(self, kb, lf):
        # Should NOT be used now.
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            tokens.append(PriceScaler.scale_price(kb, price).canonical)
        return tokens

    def _insert_markers(self, agent, utterance, new_turn):
        # Mark end of sentence
        utterance.append(markers.EOS)

        # Insert GO
        if new_turn:
            cat_symbol = category_to_marker[self.category_str]
            utterance.insert(0, cat_symbol)

            role = self.agent_to_role[agent]
            start_symbol = markers.GO_S if role == 'seller' else markers.GO_B
            utterance.insert(0, start_symbol)

        return utterance

    def _add_utterance(self, agent, utterance, lf=None, price_act=None):
        if price_act is None:
            price_act = {}

        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            new_turn = False
        else:
            new_turn = True

        utterance = self._insert_markers(agent, utterance, new_turn)
        entities = [x if is_entity(x) else None for x in utterance]

        if lf:
            pass
        else:
            print('error-lf: ', lf)
        assert lf

        self.is_int = False
        if new_turn:
            self.agents.append(agent)
            role = self.agent_to_role[agent]
            self.roles.append(role)

            self.token_turns.append(utterance)
            self.entities.append(entities)
            self.lf_turns.append(lf)
            self.modified.append(True)
            self.price_actions.append(price_act)
        else:
            self.token_turns[-1].extend(utterance)
            self.entities[-1].extend(entities)
            # self.lfs[-1].extend(lf)
            self.lf_turns.append(lf)
            print('Not new turn!')
            print(agent, utterance, lf, price_act)

            print(self.lf_turns)
            print(self.token_turns)
            print(self.agents)
            assert False

    def kb_context_to_int(self):
        self.category = self.mappings['cat_vocab'].to_ind(self.category)
        self.title = map(self.mappings['kb_vocab'].to_ind, self.title)
        self.description = map(self.mappings['kb_vocab'].to_ind, self.description)

    def lf_to_int(self):
        '''
        Used by Dialogue.convert_to_int() and NeuralSession.conver_to_int()
        lf_turns -> lfs
        token_turns -> tokens
           -> last_prices
        :return:
        '''
        last_price = [0]*2
        last_price[self.agent] = 1
        if len(self.last_prices[0]) > 0:
            last_price = [self.last_prices[i][-1] for i in range(2)]

        self.need_output = False

        for i, lf in enumerate(self.lf_turns):
            if not self.modified[i]:
                continue

            self.modified[i] = False
            self.lfs.append(lf.copy())
            self.tokens.append(self.textint_map.text_to_int(self.token_turns[i], 'encoding'))

            tmp_lf = self.lfs[-1]
            if not isinstance(tmp_lf['intent'], int):
                tmp_lf['intent'] = self.lfint_map.vocab.to_ind(tmp_lf['intent'])

            tmp_lf['original_price'] = tmp_lf.get('price')
            # Add last price
            if self.hidden_price:
                if tmp_lf.get('price') is not None:
                    last_price[self.agents[i]] = tmp_lf.get('price')

                if i >= len(self.agents):
                    print('error i{} >= len(agents){}'.format(i, len(self.agents)))
                    print(self.lf_turns)
                    print(self.token_turns)
                    print(self.agents)
                tmp_lf['price'] = last_price[self.agents[i]]

                if tmp_lf['price'] <= -1:
                    self.need_output = True

                # Update last_prices
                for agt in range(2):
                    if len(self.last_prices[agt]) <= i:
                        assert len(self.last_prices[agt]) == i
                        self.last_prices[agt].append(last_price[agt])
                    else:
                        self.last_prices[agt][-1] = last_price[agt]

            self.lfs[i] = tmp_lf

            self.rid.append(0 if self.roles[i] == 'seller' else 1)

    def convert_to_int(self):
        '''
        Used by DataGenerator, in SL part.
        '''
        if self.is_int:
            return

        self.kb_context_to_int()
        self.lf_to_int()

        for token, lf, pact in zip(self.tokens, self.lfs, self.price_actions):
            self.turns[0].append(lf)
            self.turns[1].append(token)
            self.turns[2].append(pact)

        self.is_int = True

    def _pad_list(self, l, size, pad):
        for i in range(len(l), size):
            l.append(pad)
        return l

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        * Should NOT be used in RL part.
        * pad_turns used after convert_to_int and in 'create_batches'
        '''
        if len(self.lfs) == 0:
            print('lfs is zero', self.is_int, self.modified)
            print('nums', num_turns, self.num_turns, self.num_lfs)

        # Need to pad lfs and turns by hand
        self._pad_list(self.turns[1], num_turns, [])

        # Default Settings
        pad_intent = 0
        pad_lfs = [{'intent': pad_intent, 'price': self.last_prices[0][-1]},
                   {'intent': pad_intent, 'price': self.last_prices[1][-1]}]

        self.tokens = self._pad_list(self.tokens, num_turns, [0])

        for i in range(len(self.lfs), num_turns):
            self.agents.append(self.agents[-1] ^ 1)
            self.roles.append(self.agent_to_role[self.agents[i]])
            self.rid.append(0 if self.roles[-1] == 'seller' else 1)
            self.turns[0].append(pad_lfs[self.agents[i]])
            self.lfs.append(pad_lfs[self.agents[i]])

    def get_price_turns(self, pad):
        '''
        Given flattened entity turns, return the price for each token.
        pad: used to fill in non-price targets.
        '''
        def to_float_price(entity):
            return float('{:.2f}'.format(PriceTracker.get_price(entity)))
        prices = [[to_float_price(entity) if entity else pad for entity in entities] for entities in self.entities]
        return prices

class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form, model='seq2seq'):
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                'decoding': entity_decoding_form,
                'target': entity_target_form}
        self.model = model

    @classmethod
    def get_entity_form(cls, entity, form):
        '''
        An entity is represented as (surface_form, (canonical_form, type)).
        '''
        assert len(entity) == 2
        if form == 'surface':
            return entity.surface
        elif form == 'type':
            return '<%s>' % entity.canonical.type
        elif form == 'canonical':
            if isinstance(entity, Entity):
                return entity._replace(surface='')
            else:
                return entity
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage=None):
        '''
        Input: utterance is a list of tokens, stage is either encoding, decoding or target
        Output: in most cases, stage will be declared. Based on a combination of
             the model_type and stage, we choose whether or not to summarize the
             utterance.  Models with "sum" should be summarized to only include
             selected keywords, models with "seq" will keep the full sequence.
        '''

        # Remove entities here
        if stage is None:
            ret = []
            for x in utterance:
                if not is_entity(x):
                    ret.append(x)
            return ret
        else:
            if stage == 'encoding':
                summary = self.summarize(utterance) if self.model in ["sum2sum", "sum2seq"] else utterance
            elif (stage == 'decoding') or (stage == 'target'):
                if self.model == "sum2sum":
                    summary = self.summarize(utterance)
                elif self.model == "sum2seq":
                    summary = self.summarize(utterance)
                    summary.append(markers.END_SUM)
                    summary.extend(utterance)
                else:
                    summary = utterance

            ret = []
            for x in utterance:
                if not is_entity(x):
                    ret.append(x)
            return ret

    def lf_to_tokens(self, kb, lf):
        lf = lf.copy()
        intent = lf['intent']
        if intent == 'accept':
            intent = markers.ACCEPT
        elif intent == 'reject':
            intent = markers.REJECT
        elif intent == 'quit':
            intent = markers.QUIT
        elif intent == 'offer':
            intent = markers.OFFER
        tokens = [intent]
        if lf.get('price') is not None:
            p = lf['price']
            price = Entity.from_elements(surface=p, value=p, type='price')
            price = PriceScaler.scale_price(kb, price)
            tokens.append(price)
            lf['price'] = price
        return tokens, lf

    def _process_example(self, ex):
        '''
        Convert example to turn-based dialogue from each agent's perspective
        Create two Dialogue objects for each example
        '''
        kbs = ex.scenario.kbs
        for agent in (0, 1):
            dialogue = Dialogue(agent, kbs[agent], ex.ex_id, model=self.model)
            for e in ex.events:

                if self.model in ('lf2lf',):
                    lf = e.metadata
                    utterance = self.process_event(e, dialogue.kb)
                else:
                    # No utterance
                    lf = e.metadata
                    assert lf is not None
                    utterance, lf = self.lf_to_tokens(dialogue.kb, lf)

                if utterance:
                    dialogue.add_utterance(e.agent, utterance, lf=lf)
                else:
                    print(e.data, e.metadata, e.action)
                    print('dialogue: ')
                    for i in dialogue.token_turns:
                        print(i)
                    assert False

            yield dialogue

    @classmethod
    def price_to_entity(cls, price):
        return Entity(price, CanonicalEntity(price, 'price'))

    def process_event(self, e, kb):
        '''
        Tokenize, link entities
        '''
        from cocoa.core.event import Event
        # Process semi-event
        if not isinstance(e, Event):
            if len(e) < 2 or e[1] is None:
                return e
            else:
                # print('e is:', e)
                e[1] = PriceScaler._scale_price(kb, e[1])
                return e

        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.link_entity(tokenize(e.data), kb=kb, scale=True, price_clip=4.)
            if entity_tokens:
                return entity_tokens
            else:
                return None
        elif e.action == 'offer':
            data = e.metadata['price']
            if data is None:
                return None

            entity_tokens = [markers.OFFER]
            return entity_tokens
        elif e.action == 'quit':
            entity_tokens = [markers.QUIT]
            return entity_tokens
        elif e.action == 'accept':
            entity_tokens = [markers.ACCEPT]
            return entity_tokens
        elif e.action == 'reject':
            entity_tokens = [markers.REJECT]
            return entity_tokens
        elif e.action == 'None':
            # Empty sentence
            entity_tokens = [markers.NONE]
            return entity_tokens
        else:
            raise ValueError('Unknown event action.' + str(e.action) + str(e.data) + str(e.metadata))

    # Skip examples which are too short
    @classmethod
    def skip_example(cls, example):
        tokens = {0: 0, 1: 0}
        turns = {0: 0, 1: 0}
        for event in example.events:
            if event.action == "message":
                msg_tokens = tokenize(event.data)
                tokens[event.agent] += len(msg_tokens)
                turns[event.agent] += 1
        if tokens[0] < 40 and tokens[1] < 40:
            return True
        if turns[0] < 2 or turns[1] < 2:
            return True
        return False

    def preprocess(self, examples):
        dialogues = []
        for ex in examples:
            if self.skip_example(ex):
                continue
            for d in self._process_example(ex):
                dialogues.append(d)
        return dialogues

class DataGenerator(object):
    def __init__(self, train_examples, dev_examples, test_examples, preprocessor,
            schema, mappings_path=None, cache='.cache',
            ignore_cache=False, num_context=1, batch_size=1,
            model='seq2seq',
            dia_num=0, state_length=2):
        examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.items()}
        self.num_context = num_context
        self.model = model

        self.cache = cache
        self.ignore_cache = ignore_cache
        if (not os.path.exists(cache)) or ignore_cache:
            # NOTE: each dialogue is made into two examples from each agent's perspective
            # Generate dialogue structure from examples
            self.dialogues = {k: preprocessor.preprocess(v)  for k, v in examples.items() if v}

            for fold, dialogues in self.dialogues.items():
                print('%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold]))
        else:
            self.dialogues = {k: None  for k, v in examples.items() if v}
            print('Using cached data from', cache)

        self.mappings = self.load_mappings(model, mappings_path, schema, preprocessor)
        self.textint_map = TextIntMap(self.mappings['utterance_vocab'], preprocessor)
        # if model == 'tom':
        self.lfint_map = TextIntMap(self.mappings['lf_vocab'], preprocessor)
        Dialogue.mappings = self.mappings
        Dialogue.textint_map = self.textint_map
        Dialogue.lfint_map = self.lfint_map
        Dialogue.preprocessor = preprocessor
        Dialogue.num_context = num_context

        if model == 'tom':
            Dialogue.lfint_map = self.lfint_map

        self.dialogue_batcher = DialogueBatcherFactory.get_dialogue_batcher(model,
                        kb_pad=self.mappings['kb_vocab'].to_ind(markers.PAD),
                        mappings=self.mappings, num_context=num_context,
                        dia_num=dia_num, state_length=state_length)

        from get_policy import PolicyCounter
        self.policy = PolicyCounter(len(self.mappings['lf_vocab']), from_dataset=True)
        # print(self.dialogues)
        self.policy.add_dialogues(self.dialogues)

        self.batches = {k: self.create_batches(k, dialogues, batch_size) for k, dialogues in self.dialogues.items()}


    def load_mappings(self, model_type, mappings_path, schema, preprocessor):
        vocab_path = os.path.join(mappings_path, 'vocab.pkl')
        if not os.path.exists(vocab_path):
            print('Vocab not found at', vocab_path)
            mappings = create_mappings(self.dialogues['train'], schema,
                preprocessor.entity_forms.values(), model_type)
            write_pickle(mappings, vocab_path)
            print('Wrote mappings to {}.'.format(vocab_path))
        else:
            print('Loading vocab from', vocab_path)
            mappings = read_pickle(vocab_path)

        for k, v in mappings.items():
            print(k, v.size)
        mappings = make_model_mappings(model_type, mappings)
        return mappings

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.items():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def get_dialogue_batch(self, dialogues):
        return DialogueBatcher(dialogues).create_batch()

    def dialogue_sort_score(self, d):
        # Sort dialogues by number o turns
        return -len(d.lf_turns)

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogue_batches = []
        # make sure the length of dialogue decreasing
        dialogues.sort(key=lambda d: self.dialogue_sort_score(d))
        N = len(dialogues)
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.append(self.dialogue_batcher.create_batch(dialogue_batch))
            # print(dialogue_batches[-1][0]['encoder_args']['intent'][0])
            start = end
        return dialogue_batches

    def get_all_responses(self, name):
        dialogues = self.dialogues[name]
        responses = {'seller': [], 'buyer': []}
        for dialogue in dialogues:
            for turn, role in zip(dialogue.token_turns, dialogue.roles):
                responses[role].extend(turn)
        return responses

    def create_batches(self, name, dialogues, batch_size):
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)
        cache_file = os.path.join(self.cache, '%s_batches.pkl' % name)

        # Generate Batch
        if (not os.path.exists(cache_file)) or self.ignore_cache:
            for dialogue in dialogues:
                dialogue.convert_to_int()

            dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
            print('Write %d batches to cache %s' % (len(dialogue_batches), cache_file))
            start_time = time.time()
            write_pickle(dialogue_batches, cache_file)
            print('[%d s]' % (time.time() - start_time))
        else:
            start_time = time.time()
            dialogue_batches = read_pickle(cache_file)
            print('Read %d batches from cache %s' % (len(dialogue_batches), cache_file))
            print('[%d s]' % (time.time() - start_time))
        return dialogue_batches

    def generator(self, name, shuffle=True, cuda=True):
        dialogue_batches = self.batches[name]
        #print('total number:', sum([len(b) for b in dialogue_batches]))
        yield sum([len(b) for b in dialogue_batches])
        inds = list(range(len(dialogue_batches)))
        if shuffle:
            random.shuffle(inds)
        for ind in inds:
            for batch in dialogue_batches[ind]:
                yield SLBatch(batch['encoder_args'],
                            batch['decoder_args'],
                            batch['context_data'],
                            self.mappings['lf_vocab'],
                            # num_context=self.num_context,
                            cuda=cuda,
                            # msgs=batch['msgs'],
                            )
            # End of dialogue
            yield None
