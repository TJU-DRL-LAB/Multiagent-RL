import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from .Models import RNNEncoder

class MultilayerPerceptron(nn.Module):
    """
        final_output:
            If true, last layer also have activation function.
    """
    def __init__(self, input_size, layer_size, layer_depth, final_output=None):
        super(MultilayerPerceptron, self).__init__()

        last_size = input_size
        hidden_layers = []
        if layer_depth == 0:
            self.hidden_layers = nn.Identity()
        else:
            for i in range(layer_depth):
                if final_output is not None and i == layer_depth-1:
                    hidden_layers += [nn.Linear(last_size, final_output)]
                else:
                    hidden_layers += [nn.Linear(last_size, layer_size), nn.ReLU(layer_size)]
                last_size = layer_size
            self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, input):
        return self.hidden_layers(input)


class CurrentEncoder(nn.Module):
    # intent | price | roles | number of history
    def __init__(self, input_size, embeddings, output_size, hidden_size=64, hidden_depth=2):
        super(CurrentEncoder, self).__init__()

        self.fix_emb = False

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            self.uttr_emb = embeddings
            self.uttr_lstm = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = input_size + hidden_size

        else:
            hidden_input = input_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, extra, action=None, tau=None):
        batch_size = extra.shape[0]

        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_lstm(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra], dim=-1)
        else:
            hidden_input = extra

        emb = self.hidden_layer(hidden_input)

        return emb, action, tau



class HistoryIdentity(nn.Module):

    def __init__(self, diaact_size, last_lstm_size, extra_size, identity_dim,
                 uttr_emb=None,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn'):
        super(HistoryIdentity, self).__init__()

        self.fix_emb = False
        self.identity_dim = identity_dim
        self.uttr_emb = uttr_emb


        if rnn_type == 'lstm':
            self.rnnh_number = 2
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.rnnh_number = 1
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size)

        hidden_input = last_lstm_size + extra_size

        # language part
        if uttr_emb:
            uttr_emb_size = uttr_emb.embedding_dim
            # if rnn_type == 'lstm':
            self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            # else:
            #     self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            hidden_input += hidden_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

    def _uttr_forward(self, uttr, batch_size):
        # Uttrance part
        # uttr: [tensor, tensor, ...], each tensor in shape (len_str, 1).
        uttr = uttr.copy()
        with torch.set_grad_enabled(not self.fix_emb):
            for i, u in enumerate(uttr):
                if u.dtype != torch.int64:
                    print('uttr_emb:', uttr)
                # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
            # print(uttr[i].shape)
        uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
        # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

        _, output = self.uttr_rnn(uttr)

        # For LSTM case, output=(h_1, c_1)
        if isinstance(output, tuple):
            output = output[0]

        uttr_emb = output.reshape(batch_size, -1)
        return uttr_emb

    def forward(self, diaact, extra, last_hidden, uttr=None):
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        encoder_input = [dia_emb, extra]

        if self.uttr_emb is not None:
            encoder_input.append(self._uttr_forward(uttr, batch_size))

        hidden_input = torch.cat(encoder_input, dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryEncoder(nn.Module):
    """RNN Encoder

    """
    def __init__(self, diaact_size, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True):
        super(HistoryEncoder, self).__init__()

        last_lstm_size = hidden_size

        if rnn_type == 'lstm':
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size, nonlinearity='relu')

        # hidden_input = last_lstm_size + extra_size
        #
        # self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False

        self.uttr_emb = embeddings

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            if rnn_type == 'lstm':
                self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            else:
                self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = hidden_size + last_lstm_size + extra_size
        else:
            hidden_input = last_lstm_size + extra_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, state, identity_state):
        # State Encoder
        diaact, extra, last_hidden = identity_state
        # identity, next_hidden = self.identity(*identity_state)
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        # Uttr Encoder
        batch_size = state.shape[0]
        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra, dia_emb], dim=-1)
        else:

            hidden_input = torch.cat([extra, dia_emb], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryIDEncoder(nn.Module):
    """
        ID move to the last layer
        Final output is [hidden, identity], so the size of hidden is (output_size-identity_size).
    """
    def __init__(self, identity, state_size, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True, rnn_state=False):
        super(HistoryIDEncoder, self).__init__()

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False
        self.rnn_state = rnn_state
        # For split input rnn hidden
        self.id_rnnh_number = 0
        self.state_rnnh_number = 0

        self.identity = identity
        self.uttr_emb = embeddings
        hidden_input = extra_size

        if identity:
            identity_size = identity.identity_dim
        else:
            identity_size = 0

        if rnn_state:
            self.state_rnnh_number = 1
            last_lstm_size = hidden_size
            if rnn_type == 'lstm':
                self.dia_rnn = torch.nn.LSTMCell(input_size=state_size, hidden_size=last_lstm_size)
            else:
                self.dia_rnn = torch.nn.RNNCell(input_size=state_size, hidden_size=last_lstm_size, nonlinearity='relu')
            hidden_input += last_lstm_size
        else:
            hidden_input += state_size


        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            # if rnn_type == 'lstm':
            self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            # else:
            #     self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input += hidden_size

        if identity:
            self.id_rnnh_number = self.identity.rnnh_number

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size - identity_size, hidden_depth)

    def forward(self, uttr, dia_act, state, extra, rnn_hiddens, id_gt=None):
        encoder_input = [extra]
        next_rnnh = ()
        batch_size = dia_act.shape[0]

        # split rnn_hiddens
        if not isinstance(rnn_hiddens, tuple):
            rnn_hiddens = (rnn_hiddens,)
        id_rnnh = rnn_hiddens[-self.id_rnnh_number:]
        state_rnnh = rnn_hiddens[:self.state_rnnh_number]
        if self.id_rnnh_number == 1: id_rnnh = id_rnnh[0]
        if self.state_rnnh_number == 1: state_rnnh = state_rnnh[0]

        # State Part
        if self.rnn_state:
            next_hidden = self.dia_rnn(dia_act, state_rnnh)
            if isinstance(next_hidden, tuple):
                # For LSTM
                dia_emb = next_hidden[0].reshape(batch_size, -1)
                next_rnnh = next_hidden
            else:
                # For RNN
                dia_emb = next_hidden.reshape(batch_size, -1)
                next_rnnh = (next_hidden,)
            encoder_input.append(dia_emb)
        else:
            encoder_input.append(state)

        # Uttrance part
        if self.uttr_emb is not None:
            # uttr: [tensor, tensor, ...], each tensor in shape (len_str, 1).
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)
            encoder_input.append(uttr_emb)

        # Main part
        hidden_input = torch.cat(encoder_input, dim=-1)
        emb = self.hidden_layer(hidden_input)

        # Identity part
        identity = None
        if self.identity:
            if id_gt is not None:
                identity = id_gt
                _identity = id_gt
                next_rnnh = next_rnnh + (torch.zeros_like(id_gt),)
            else:
                identity, next_hidden = self.identity(dia_act, extra, id_rnnh, uttr)
                if isinstance(next_hidden, tuple): next_rnnh = next_rnnh + next_hidden
                else: next_rnnh = next_rnnh + (next_hidden,)

                if self.fix_identity:
                    _identity = identity.detach()
                else:
                    _identity = identity
                if self.ban_identity:
                    _identity.fill_(0)
                _identity = torch.softmax(_identity, dim=1)
            emb = torch.cat([emb, _identity], dim=-1)

        return emb, next_rnnh, identity

class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            embedding_size=64,
            num_quantiles=32,
            layer_norm=True,
            **kwargs
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = torch.from_numpy(np.arange(1, 1 + self.embedding_size)).float()

    def forward(self, encoder_output):
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)
        """

        state, action, tau = encoder_output


        device = tau.device

        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec.to(device) * np.pi)  # (N, T, E)
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output

class SinglePolicy(nn.Module):

    def __init__(self, input_size, output_size, hidden_depth=1, hidden_size=128, ):
        super(SinglePolicy, self).__init__()

        self.hidden_layers = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, emb):
        if isinstance(emb, tuple):
            emb = emb[0]

        hidden_state = self.hidden_layers(emb)
        output = self.output_layer(hidden_state)
        return output

        
# For RL Agent
# Intent + Price(action)
class MixedPolicy(nn.Module):

    def __init__(self, input_size, intent_size, price_size, hidden_size=64, hidden_depth=2, price_extra=0):
        super(MixedPolicy, self).__init__()
        self.common_net = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
        self.intent_net = SinglePolicy(hidden_size, intent_size, hidden_depth=1, hidden_size=hidden_size)
        self.price_net = SinglePolicy(hidden_size + price_extra, price_size, hidden_depth=1, hidden_size=hidden_size//2)
        self.intent_size = intent_size
        self.price_size = price_size

    def forward(self, encoder_output, price_extra=None):
        state_emb = encoder_output[0]
        common_state = self.common_net(state_emb)

        intent_output = self.intent_net(common_state)

        price_input = [common_state]
        if price_extra:
            price_input.append(price_extra)
        price_input = torch.cat(price_input, dim=-1)
        price_output = self.price_net(price_input)

        return intent_output, price_output


class CurrentModel(nn.Module):

    def __init__(self, encoder, decoder, fix_encoder=False):
        super(CurrentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            encoder_output = self.encoder(*input)
        output = self.decoder(encoder_output)
        return output


class HistoryModel(nn.Module):

    def __init__(self, encoder, decoder, fix_encoder=False):
        super(HistoryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder
        self.hidden_vec = None
        self.hidden_stra = None

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            # return emb, next_hidden, (identity)
            e_output = self.encoder(*input)

        if e_output[-1] is not None:
            self.hidden_vec = e_output[-1].cpu().data.numpy()
        else:
            self.hidden_vec = None

        d_output = self.decoder(e_output[0])
        return (d_output,) + e_output[1:]
