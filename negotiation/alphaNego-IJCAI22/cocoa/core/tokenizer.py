import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.tokenize.moses import MosesDetokenizer

detokenizer = MosesDetokenizer()

def detokenize(tokens):
    return detokenizer.detokenize(tokens, return_str=True)

def tokenize(utterance, lowercase=True):
    if lowercase:
        utterance = utterance.lower()
    tokens = word_tokenize(utterance)
    return tokens
