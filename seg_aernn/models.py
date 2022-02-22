"""
Segmental autoencoding recurrent neural network (segmental AE-RNN) modules.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class InputDropout(nn.Module):
    """
    Based on:
    - https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py
    - https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        x = x.clone()
        mask = x.new_empty(
            x.size(0), x.size(1), requires_grad=False
            ).bernoulli_(1 - self.p)
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(2))
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        
        return x*mask


class Encoder(nn.Module):
    """Encodes a sequence of symbol embeddings into a single embedding."""
    
    def __init__(self, n_symbols, symbol_embedding_dim, hidden_dim,
            embedding_dim, n_layers=1, dropout=0., input_dropout=0,
            bidirectional=False):
        super(Encoder, self).__init__()
        self.bidirectional=bidirectional
        self.embedding = nn.Embedding(n_symbols, symbol_embedding_dim)
        self.rnn = nn.GRU(
            symbol_embedding_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout, bidirectional=self.bidirectional
            )
        self.fc = nn.Linear(
            hidden_dim*2 if self.bidirectional else hidden_dim, embedding_dim
            )
        self.input_dropout = InputDropout(p=input_dropout)
        
    def forward(self, x, lengths):
        """
        Parameters
        ----------        
        x : Tensor [n_batch, length, dim]
        """
        x = self.embedding(x)
        x = self.input_dropout(x)
        
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
            )
        rnn_hidden, final_hidden = self.rnn(packed)
        if self.bidirectional:
            final_hidden = torch.cat(
                [final_hidden[0, :, :], final_hidden[1, :, :]], dim=1
                ).unsqueeze(0)
        output = self.fc(final_hidden[-1])
        return rnn_hidden, output 



class Decoder1(nn.Module):
    """A decoder conditioned by setting its first hidden state."""
    
    def __init__(self, n_symbols, symbol_embedding_dim, embedding_dim,
            hidden_dim, sos_id, n_layers=1, dropout=0.,
            teacher_forcing_ratio=1):
        super(Decoder1, self).__init__()
        self.sos_id = sos_id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.embedding = nn.Embedding(n_symbols, symbol_embedding_dim)
        self.bridge = nn.Linear(
            embedding_dim, hidden_dim*n_layers, bias=True
            )
        self.rnn = nn.GRU(
            symbol_embedding_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout
            )
        self.fc = nn.Linear(hidden_dim, n_symbols, bias=False)
    
    def forward(self, encoder_embedding, y, lengths):
        max_length = max(lengths)
        
        # Initialise the first decoder hidden state
        hidden = self.bridge(encoder_embedding)
        hidden = hidden.unsqueeze(0)  # [n_layers, batch, dim]
        if self.rnn.num_layers > 1:
            split_hidden = []
            for i_layer in range(self.rnn.num_layers):
                split_hidden.append(
                    hidden[:, :,
                    i_layer*self.rnn.hidden_size:(i_layer +
                    1)*self.rnn.hidden_size]
                    )
            hidden = torch.cat(split_hidden, dim=0)  # [n_layers, batch, dim]
        
        # Decoder RNN hidden states
        rnn_hidden = []
        output_list = []
        
        # Unroll the decoder RNN for max_length steps
        prev_y = torch.ones(
            encoder_embedding.shape[0], 1
            ).fill_(self.sos_id).type_as(y)  # first input is <s>
        for i in range(max_length):
            if (np.random.random() < self.teacher_forcing_ratio and not (i ==
                    0)):
                prev_embedding = self.embedding(y[:, i - 1]).unsqueeze(1)
                # [n_batch, length, dim]
            else:
                prev_embedding = self.embedding(prev_y)
            output, hidden = self.rnn(prev_embedding, hidden)
            rnn_hidden.append(output)
            prob_output = F.log_softmax(self.fc(output), dim=-1)
            output_list.append(prob_output)
            _, prev_y = torch.max(prob_output, dim=-1)

        rnn_hidden = torch.cat(rnn_hidden, 1)
        output_list = torch.cat(output_list, 1)
        return rnn_hidden, output_list

    def greedy_decode(self, encoder_embedding, max_length):
        """
        Performs greedy decoding given the encoder embedding.
        
        Parameters
        ----------
        encoder_embedding : Tensor [n_batch, dim]
        """

        # Initialise the first decoder hidden state
        hidden = self.bridge(encoder_embedding)
        hidden = hidden.unsqueeze(0)  # [n_layers, batch, dim]
        if self.rnn.num_layers > 1:
            split_hidden = []
            for i_layer in range(self.rnn.num_layers):
                split_hidden.append(
                    hidden[:, :,
                    i_layer*self.rnn.hidden_size:(i_layer +
                    1)*self.rnn.hidden_size]
                    )
            hidden = torch.cat(split_hidden, dim=0)  # [n_layers, batch, dim]

        # Unroll the decoder RNN for max_length steps
        prev_y = torch.ones(
            encoder_embedding.shape[0], 1).fill_(
            self.sos_id
            ).type(torch.LongTensor).to(encoder_embedding.get_device()
            )
        output_sequence = []
        output_log_prob = []
        for i in range(max_length):
            prev_embedding = self.embedding(prev_y)
            # [n_batch, length, dim]
            output, hidden = self.rnn(prev_embedding, hidden)
            prob_output = F.log_softmax(self.fc(output), dim=-1)
            next_log_prob, next_y = torch.max(prob_output, dim=-1)
            output_sequence.append(next_y.cpu().numpy())
            output_log_prob.append(next_log_prob.cpu().numpy())
            prev_y = next_y
        
        output_sequence = np.array(output_sequence).squeeze().T 
        output_log_prob = np.array(output_log_prob).squeeze().T 
        # [n_batch, length]

        return output_sequence, output_log_prob


class Decoder2(nn.Module):
    """A non-autoregressive decoder conditioned at each time step."""
    
    def __init__(self, n_symbols, embedding_dim, hidden_dim,
            n_layers=1, dropout=0.):
        super(Decoder2, self).__init__()
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, n_layers, batch_first=True,
            dropout=dropout
            )
        self.fc = nn.Linear(hidden_dim, n_symbols, bias=False)
    
    def forward(self, encoder_embedding, y, lengths):
        max_length = max(lengths)
               
        # Decoder RNN hidden states
        rnn_hidden = []
        output_list = []
        
        # Unroll the decoder RNN for max_length steps
        hidden = None
        for i in range(max_length):
            if hidden is None:
                output, hidden = self.rnn(encoder_embedding.unsqueeze(1))
            else:
                output, hidden = self.rnn(
                    encoder_embedding.unsqueeze(1), hidden
                    )
            rnn_hidden.append(output)
            prob_output = F.log_softmax(self.fc(output), dim=-1)
            output_list.append(prob_output)

        rnn_hidden = torch.cat(rnn_hidden, 1)
        output_list = torch.cat(output_list, 1)
        return rnn_hidden, output_list

    def greedy_decode(self, encoder_embedding, max_length):
        """
        Performs greedy decoding given the encoder embedding.
        
        Parameters
        ----------
        encoder_embedding : Tensor [n_batch, dim]
        """

        hidden = None
        output_sequence = []
        output_log_prob = []
        for i in range(max_length):
            if hidden is None:
                output, hidden = self.rnn(encoder_embedding.unsqueeze(1))
            else:
                output, hidden = self.rnn(
                    encoder_embedding.unsqueeze(1), hidden
                    )
            prob_output = F.log_softmax(self.fc(output), dim=-1)
            next_log_prob, next_y = torch.max(prob_output, dim=-1)
            output_sequence.append(next_y.cpu().numpy())
            output_log_prob.append(next_log_prob.cpu().numpy())

        output_sequence = np.array(output_sequence).squeeze().T
        output_log_prob = np.array(output_log_prob).squeeze().T

        return output_sequence, output_log_prob


class EncoderDecoder(nn.Module):
    """A encoder-decoder."""

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, x_lengths, y, y_lengths):
        """
        Parameters
        ----------
        x : Tensor [n_batch, length, dim]
        """
        encoder_rnn, encoder_embedding = self.encoder(x, x_lengths)
        decoder_rnn, decoder_output = self.decoder(
            encoder_embedding, y, y_lengths
            )
        return encoder_embedding, decoder_output
