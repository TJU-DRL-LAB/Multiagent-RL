import random
import re
import numpy as np
import torch
from onmt.Utils import use_gpu

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity

from core.event import Event
from .session import Session
from neural.preprocess import markers, Dialogue
import time
from neural.batcher_rl import ToMBatch
from sessions.neural_session import PytorchNeuralSession


class PytorchNeuralTomSession(PytorchNeuralSession):
    def send(self, temperature=1, is_fake=False, strategy=-1):
        # not fake: update tom_hidden
        # fake: return policy
        tokens, output_data = self.generate(is_fake=is_fake, hidden_state=self.tom_hidden, strategy=[strategy])
        if is_fake:
            tmp_time = time.time()
            return self.try_all_aa(tokens, output_data)
        else:
            self.tom_hidden = output_data['rnn_hidden']

    def generate(self, temperature=1, is_fake=False, acpt_range=None, hidden_state=None, strategy=None):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
        batch = self._create_batch()
        tom_batch = ToMBatch.from_raw(batch, strategy)

        intents, prices = batch.get_pre_info(self.lf_vocab)
        last_prices = [prices[0, 0].item(), prices[0, 1].item()]

        output_data = self.generator.generate_batch(tom_batch, enc_state=None, whole_policy=is_fake,
                                                    hidden_state=hidden_state)

        entity_tokens = self._output_to_tokens(output_data)
        output_data['last_prices'] = last_prices

        return entity_tokens, output_data