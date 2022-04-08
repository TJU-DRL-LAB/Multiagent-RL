import math
import re
from collections import defaultdict
from itertools import chain

from cocoa.core.entity import Entity, CanonicalEntity
from cocoa.core.util import read_json, write_pickle, read_pickle

from core.tokenizer import tokenize


class PriceList(object):

    pList = None

    @classmethod
    def getPriceList(cls):
        if cls.pList is None:
            cls.pList = PriceList()
        return cls.pList

    def __init__(self, max_bound=3, mini_step=0.05):
        self.p_list = [-max_bound, +max_bound]
        start, end = -max_bound/2, +max_bound/2
        while start <= end:
            self.p_list.append(start)
            start += mini_step
        self.p_list = [float('{:.2f}'.format(p)) for p in self.p_list]
        self.p_list = sorted(self.p_list)

    def get_round(self, number):
        if number <= self.p_list[0]:
            return self.p_list[0]
        if number >= self.p_list[-1]:
            return self.p_list[-1]
        for i, a in enumerate(self.p_list):
            if (a <= number) and (number <= self.p_list[i+1]):
                return a if number <= (self.p_list[i+1]+a)/2 else self.p_list[i+1]
        return None

class PriceScaler(object):
    @classmethod
    def get_price_range(cls, kb):
        '''
        Return the bottomline and the target
        '''
        t = kb.facts['personal']['Target']  # 1
        role = kb.facts['personal']['Role']

        if role == 'seller':
            b = t * 0.7
            # print('[Role: {}]\ttarget: {},\tbottom {}'.format(role, t, b))
            # Augment the price range
            # b = b * 0.5
            # t = t * 2
            # unit = t-b
            # t = t+unit*2
            # b = b-unit
        else:
            b = kb.facts['item']['Price']
            # print('[Role: {}]\tbottom {},\ttarget: {}\tratio:{}'.format(role, b, t, 1.*t/b))
            # Augment the price range
            # unit = t-b
            # t = t+unit*0.5
            # b = b-unit


        return b, t

    @classmethod
    def get_parameters(cls, b, t):
        '''
        Return (slope, constant) parameters of the linear mapping.
        '''
        assert (t - b) != 0
        w = 1. / (t - b)
        c = -1. * b / (t - b)
        return w, c

    @classmethod
    # TODO: this is operated on canonical entities, need to be consistent!
    # INPUT: a scale form 0 to 1
    # OUTPUT: price is entity with real price
    def unscale_price(cls, kb, price):
        p = PriceTracker.get_price(price)
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        assert w != 0
        p = (p - c) / w
        p = round(p)
        if isinstance(price, Entity):
            return price._replace(canonical=price.canonical._replace(value=p))
        elif isinstance(price, CanonicalEntity):
            return price._replace(value=p)
        else:
            return p

    @classmethod
    def _scale_price(cls, kb, p):
        b, t = cls.get_price_range(kb)
        w, c = cls.get_parameters(b, t)
        p = w * p + c
        # Discretize to two digits
        #p = float('{:.2f}'.format(p))
        # p = PriceList.getPriceList().get_round(p)
        # print("scale_result:{}".format(p))
        return p

    # INPUT: real price (float number)
    # OUTPUT: price is a float number in [0, 1]
    @classmethod
    def scale_price(cls, kb, price):
        """Scale the price such that bottomline=0 and target=1.

        Args:
            price (float)
        """
        # p = PriceTracker.get_price(price)
        p = cls._scale_price(kb, price)
        # return price._replace(canonical=price.canonical._replace(value=p))
        return p

class PriceTracker(object):
    def __init__(self, model_path):
        self.model = read_pickle(model_path)

    @classmethod
    def get_price(cls, token):
        # print('token', token)
        if isinstance(token, Entity):
            return token.canonical.value
        elif isinstance(token, CanonicalEntity):
            return token.value
        elif isinstance(token, float):
            return token
        else:
            return token

    @classmethod
    def process_string(cls, token):
        token = re.sub(r'[\$\,]', '', token)
        try:
            if token.endswith('k'):
                token = str(float(token.replace('k', '')) * 1000)
        except ValueError:
            pass
        return token

    def is_price(self, left_context, right_context):
        if left_context in self.model['left'] and right_context in self.model['right']:
            return True
        else:
            return False

    def get_kb_numbers(self, kb):
        title = tokenize(re.sub(r'[^\w0-9\.,]', ' ', kb.facts['item']['Title']))
        description = tokenize(re.sub(r'[^\w0-9\.,]', ' ', ' '.join(kb.facts['item']['Description'])))
        numbers = set()
        for token in chain(title, description):
            try:
                numbers.add(float(self.process_string(token)))
            except ValueError:
                continue
        return numbers

    def link_entity(self, raw_tokens, kb=None, scale=True, price_clip=None):
        tokens = ['<s>'] + raw_tokens + ['</s>']
        entity_tokens = []
        if kb:
            kb_numbers = self.get_kb_numbers(kb)
            list_price = kb.facts['item']['Price']
        for i in range(1, len(tokens)-1):
            token = tokens[i]
            try:
                number = float(self.process_string(token))
                has_dollar = lambda token: token[0] == '$' or token[-1] == '$'
                # Check context
                if not has_dollar(token) and \
                        not self.is_price(tokens[i-1], tokens[i+1]):
                    number = None
                # Avoid 'infinity' being recognized as a number
                elif number == float('inf') or number == float('-inf'):
                    number = None
                # Check if the price is reasonable
                elif kb:
                    if not has_dollar(token):
                        if number > 1.5 * list_price:
                            number = None
                        # Probably a spec number
                        if number != list_price and number in kb_numbers:
                            number = None
                    if number is not None and price_clip is not None:
                        scaled_price = PriceScaler._scale_price(kb, number)
                        # TODO: less than 10%?
                        if abs(scaled_price) > price_clip:
                            number = None
            except ValueError:
                number = None
            if number is None:
                new_token = token
            else:
                assert not math.isnan(number)
                if scale:
                    scaled_price = PriceScaler._scale_price(kb, number)
                else:
                    scaled_price = number
                new_token = Entity(surface=token, canonical=CanonicalEntity(value=scaled_price, type='price'))
            entity_tokens.append(new_token)
        return entity_tokens

    @classmethod
    def train(cls, examples, output_path=None):
        '''
        examples: json chats
        Use "$xxx$ as ground truth, and record n-gram context before and after the price.
        '''
        context = {'left': defaultdict(int), 'right': defaultdict(int)}
        for ex in examples:
            for event in ex['events']:
                if event['action'] == 'message':
                    tokens = tokenize(event['data'])
                    tokens = ['<s>'] + tokens + ['</s>']
                    for i, token in enumerate(tokens):
                        if token[0] == '$' or token[-1] == '$':
                            context['left'][tokens[i-1]] += 1
                            context['right'][tokens[i+1]] += 1
        if output_path:
            write_pickle(context, output_path)
        return context

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-examples-path', help='Path to training json file')
    parser.add_argument('--output', help='Path to output model')
    args = parser.parse_args()

    examples = read_json(args.train_examples_path)
    PriceTracker.train(examples, args.output)
