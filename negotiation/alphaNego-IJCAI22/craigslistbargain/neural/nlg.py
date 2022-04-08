## NLG module

import json
import difflib
# from termcolor import colored
import string
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import numpy as np
import random
from cocoa.core.entity import is_entity, Entity, CanonicalEntity

class IRNLG(object):
    def __init__(self, args):
        self.gen_dic = {}
        with open(args.nlg_dir) as json_file:  
            self.gen_dic = json.load(json_file)

    def _add_strategy_in_uttr(self, uttr, stra):
        # c = self.env.vocab.size - 1 - self.price_strategy_label
        uttr = uttr.copy()
        # for each sentences
        if random.randint(0, 5) > 0:
            return uttr
        uttr.insert(random.randint(0, len(uttr)), stra)
        return uttr

    def gen(self, lf, role, category, as_tokens=False, add_stra=None):
        if self.gen_dic[category].get(lf.get('intent')) is None:
            # print('not in nlg:', lf, role, category)
            new_words = ['']
            if add_stra is not None:
                new_words = self._add_strategy_in_uttr(new_words, add_stra)
            if not as_tokens:
                new_words = "".join(new_words)
            return new_words, (lf.get('intent'), role, category, 0)

        tid = random.randint(0, len(self.gen_dic[category][lf.get('intent')][role])-1)
        template = self.gen_dic[category][lf.get('intent')][role][tid]
        words = word_tokenize(template)
        new_words = []
        for i, wd in enumerate(words):
            if wd == "PPRRIICCEE" and lf.get('price'):
                if as_tokens:
                    new_words.append(CanonicalEntity(type='price', value=lf.get('price')))
                else:
                    new_words.append('$'+str(lf.get('price')))
            else:
                new_words.append(wd)

        if add_stra is not None:
            new_words = self._add_strategy_in_uttr(new_words, add_stra)

        # TODO: raw uttrence
        if as_tokens:
            return new_words, (lf.get('intent'), role, category, tid)

        sentence = "".join([" "+i if not i.startswith("'") and i not in string.punctuation
                        else i for i in new_words]).strip()

        return sentence, (lf.get('intent'), role, category, tid)