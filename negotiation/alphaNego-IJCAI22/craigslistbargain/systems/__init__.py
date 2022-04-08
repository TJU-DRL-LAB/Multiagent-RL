from cocoa.core.util import read_json, read_pickle

import options


def get_system(name, args, schema=None, timed=False, model_path=None, id=0):
    from core.price_tracker import PriceTracker
    lexicon = PriceTracker(args.price_tracker_model)

    if name in ['pt-neural', 'pt-neural-r', 'pt-neural-s']:
        from .neural_system import PytorchNeuralSystem
        # assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed, name=name, id=id)
    elif name == 'tom':
        from .neural_system import PytorchNeuralSystem
        assert model_path
        return PytorchNeuralSystem(args, schema, lexicon, model_path, timed, name=name, id=id)
    elif name == 'pt-neural-dsac':
        from .dsac_neural_system import DSACNeuralSystem
        return DSACNeuralSystem(args, schema, lexicon, model_path, timed, name=name, id=id)
    else:
        raise ValueError('Unknown system %s' % name)
