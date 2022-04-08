import random
import ujson as json
import string
import pickle
import numpy as np
import os

def random_multinomial(probs):
    target = random.random()
    i = 0
    accum = 0
    while True:
        accum += probs[i]
        if accum >= target:
            return i
        i += 1

def generate_uuid(prefix):
    return prefix + '_' + ''.join([random.choice(string.digits + string.ascii_letters) for _ in range(16)])

def read_json(path):
    return json.load(open(path))

def write_json(raw, path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'w') as out:
        print(json.dumps(raw), end='', file=out)

def read_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def write_pickle(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)

def normalize(a):
    ma = np.max(a)
    mi = np.min(a)
    assert ma > mi
    a = (a - mi) / (ma - mi)
    return a
