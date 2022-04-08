from cocoa.model.vocab import Vocabulary
from cocoa.neural.vocab_builder import build_utterance_vocab
from cocoa.core.entity import Entity, CanonicalEntity, is_entity

from .symbols import markers, sequence_markers, category_markers

def build_kb_vocab(dialogues, special_symbols=[]):
    kb_vocab = Vocabulary(offset=0, unk=True)
    cat_vocab = Vocabulary(offset=0, unk=False)

    for dialogue in dialogues:
        assert dialogue.is_int is False
        kb_vocab.add_words(dialogue.title)
        kb_vocab.add_words(dialogue.description)
        cat_vocab.add_word(dialogue.category)

    kb_vocab.add_words(special_symbols, special=True)
    kb_vocab.finish(freq_threshold=5)
    cat_vocab.add_words(['bike', 'car', 'electronics', 'furniture', 'housing', 'phone'], special=True)
    cat_vocab.finish()

    print('KB vocab size:', kb_vocab.size)
    print('Category vocab size:', cat_vocab.size)
    return kb_vocab, cat_vocab


def build_lf_vocab_simple(dialogues):
    print('[Warning] unknown function used.')
    vocab = Vocabulary(offset=0, unk=True)
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for lf in dialogue.lf_turns:
           vocab.add_words(lf['intent'])
    vocab.add_words(sequence_markers, special=True)
    print('LF vocabulary size:', vocab.size)
    return vocab


def get_entity_form(entity, form):
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


def build_lf_vocab(dialogues, special_symbols=[], entity_forms=[], except_words=[]):
    vocab = Vocabulary(offset=0, unk=True, except_words=except_words)

    def _add_entity(entity):
        for entity_form in entity_forms:
            word = get_entity_form(entity, entity_form)
            vocab.add_word(word)

    for dialogue in dialogues:
        assert dialogue.is_int is False
        for lf in dialogue.lf_turns:
            # for token in lf:
            # if is_entity(lf):
            #     _add_entity(token)
            # else:
            vocab.add_word(lf['intent'])
    vocab.add_words(sequence_markers, special=True)
    vocab.finish(size_threshold=10000)
    print('LF vocabulary size:', vocab.size)
    return vocab


def create_mappings(dialogues, schema, entity_forms, model_type, only_act=True):

    # remove all useless mapping
    except_words = category_markers + sequence_markers
    except_words.remove(markers.PAD)

    uttter_except_words = category_markers + sequence_markers

    utterance_vocab = build_utterance_vocab(dialogues, sequence_markers, entity_forms, except_words=uttter_except_words)
    kb_vocab, cat_vocab = build_kb_vocab(dialogues, [markers.PAD])
    if model_type == "tom":
        lf_vocab = build_lf_vocab(dialogues, sequence_markers, entity_forms, except_words=except_words)
    else:
        # lf_vocab = build_lf_vocab_simple(dialogues)
        lf_vocab = build_lf_vocab(dialogues, sequence_markers, entity_forms, except_words=except_words)

    return {'utterance_vocab': utterance_vocab,
            'kb_vocab': kb_vocab,
            'cat_vocab': cat_vocab,
            'lf_vocab': lf_vocab,
            }
