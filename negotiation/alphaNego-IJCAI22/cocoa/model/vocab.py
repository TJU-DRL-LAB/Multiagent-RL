import numpy as np
import time
from collections import Counter

class Vocabulary(object):

    # UNK = '<unk>'
    # TODO: Pay attenaion! We used luis data, so change the unkown marker.
    UNK = '<unknown>'
    PAD = '<pad>'

    def __init__(self, offset=0, unk=True, max_bound=3, mini_step=0.01, discrete_price=False, except_words=[]):
        self.word_to_ind = {}
        self.ind_to_word = {}
        self.word_count = Counter()
        self.size = 0
        self.offset = offset
        self.special_words = set()
        self.finished = False

        self.discrete_price = discrete_price
        self.except_words = except_words
        if discrete_price:
            self._init_prices()

        self.unk = False
        if unk:
            self.unk = True
            self.add_word(self.UNK, special=True)

    def _init_prices(self):
        from core.price_tracker import PriceList
        self.p_list = PriceList.getPriceList().p_list
        from cocoa.core.entity import Entity, CanonicalEntity
        self.add_words([Entity(surface='', canonical=CanonicalEntity(value=p, type='price')) for p in self.p_list])

    def __len__(self):
        return self.size

    def add_words(self, words, special=False):
        for w in words:
            self.add_word(w, special)

    def has(self, word):
        return word in self.word_to_ind

    def add_word(self, word, special=False):
        from cocoa.core.entity import Entity, CanonicalEntity
        if word in self.except_words:
            return

        if (isinstance(word, Entity) or isinstance(word, CanonicalEntity))and not self.discrete_price:
            return

        self.word_count[word] += 1
        if special:
            print('add', word)
            self.special_words.add(word)

    def finish(self, freq_threshold=0, size_threshold=None):
        # Let <pad> be index 0
        self.word_to_ind = {self.PAD: 0}
        self.ind_to_word = [self.PAD]

        # Make sure special words are included
        n = len(self.ind_to_word)
        for w in self.special_words:
            if w not in self.word_to_ind:
                self.ind_to_word.append(w)
                self.word_to_ind[w] = n
                n += 1

        # Add other words with threshold
        if freq_threshold > 0:
            self.word_count = {word: count for word, count in self.word_count.items()\
                               if count < freq_threshold}
        self.word_count = Counter(self.word_count)
        old_wti = self.word_to_ind.copy()
        for w, c in self.word_count.most_common(size_threshold):
            if w not in old_wti:
                self.word_to_ind[w] = n
                self.ind_to_word.append(w)
                n += 1

        self.size = len(self.ind_to_word)

        self.finished = True

    def to_ind(self, word):
        from cocoa.core.entity import Entity, CanonicalEntity
        if word is None:
            return word

        if not self.discrete_price and isinstance(word, Entity):
            # Use float value as price
            return word.canonical.value

        if not self.discrete_price and isinstance(word, CanonicalEntity):
            # Use float value as price
            return word.value

        ind = self.word_to_ind.get(word)
        if ind is None:
            ind = self.word_to_ind.get(self.UNK)
        if ind is not None:
            return ind
        else:
            print(self.ind_to_word)
            raise KeyError(str(word))

        if word in self.word_to_ind:
            return self.word_to_ind[word]
        else:
            # NOTE: if UNK is not enabled, it will throw an exception
            if self.UNK in self.word_to_ind:
                return self.word_to_ind[self.UNK]
            else:
                raise KeyError(str(word))

    def to_word(self, ind):
        if isinstance(ind, int):
            if ind >= len(self.ind_to_word):
                ind = ind - 1
            return self.ind_to_word[ind]
        else:
            from cocoa.core.entity import CanonicalEntity
            return CanonicalEntity(value=ind, type='price')

    def dump(self):
        for i, w in enumerate(self.ind_to_word):
            print('{:<8}{:<}'.format(i, w))
            if i > 100:
                break

    def load_embeddings(self, wordvec_file, dim):
        print('Loading pretrained word vectors:', wordvec_file)
        start_time = time.time()
        embeddings = np.random.uniform(-1., 1., [self.size, dim])
        num_exist = 0
        with open(wordvec_file, 'r') as f:
            for line in f:
                ss = line.split()
                word = ss[0]
                if word in self.word_to_ind:
                    num_exist += 1
                    vec = np.array([float(x) for x in ss[1:]])
                    embeddings[self.word_to_ind[word]] = vec
        print('[%d s]' % (time.time() - start_time))
        print('%d pretrained' % num_exist)
        return embeddings
