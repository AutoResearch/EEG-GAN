import math
import os
import random
import warnings
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor


# from utils.get_filter import moving_average as filter

class Autoencoder(nn.Module):

    TARGET_CHANNELS = 0
    TARGET_TIMESERIES = 1
    TARGET_BOTH = 2

    def __init__(self, input_dim: int, output_dim: int, output_dim_2: int, hidden_dim: int, target: int, num_layers=3, dropout=0.1, activation='linear', **kwargs):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_2 = output_dim_2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.target = target
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation function of type '{activation}' was not recognized.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder block of linear layers constructed in a loop and passed to a sequential container
        encoder_block = []
        encoder_block.append(nn.Linear(input_dim, hidden_dim))
        encoder_block.append(nn.Dropout(dropout))
        # encoder_block.append(self.activation)
        encoder_block.append(nn.Tanh())
        for i in range(num_layers):
            encoder_block.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_block.append(nn.Dropout(dropout))
            # encoder_block.append(self.activation)
            encoder_block.append(nn.Tanh())
        encoder_block.append(nn.Linear(hidden_dim, output_dim))
        # encoder_block.append(self.activation)
        encoder_block.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_block)

        # decoder block of linear layers constructed in a loop and passed to a sequential container
        decoder_block = []
        decoder_block.append(nn.Linear(output_dim, hidden_dim))
        decoder_block.append(nn.Dropout(dropout))
        # decoder_block.append(self.activation)
        decoder_block.append(nn.Tanh())
        for i in range(num_layers):
            decoder_block.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_block.append(nn.Dropout(dropout))
            # decoder_block.append(self.activation)
            decoder_block.append(nn.Tanh())
        decoder_block.append(nn.Linear(hidden_dim, input_dim))
        decoder_block.append(self.activation)
        self.decoder = nn.Sequential(*decoder_block)

    def forward(self, x):
        encoded = self.encoder(x.to(self.device))
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, data):
        if self.target == self.TARGET_TIMESERIES:
            data = data.permute(0, 2, 1)
        data = self.encoder(data.to(self.device))
        if self.target == self.TARGET_TIMESERIES:
            data = data.permute(0, 2, 1)
        return data

    def decode(self, encoded):
        if self.target == self.TARGET_TIMESERIES:
            encoded = encoded.permute(0, 2, 1)
        data = self.decoder(encoded)
        if self.target == self.TARGET_TIMESERIES:
            data = data.permute(0, 2, 1)
        return data


class TransformerAutoencoder(Autoencoder):

    def __init__(self, input_dim: int, output_dim: int, output_dim_2: int, target: int, hidden_dim=256, num_layers=3, num_heads=4, dropout=0.1, activation='linear', **kwargs):
        super(TransformerAutoencoder, self).__init__(input_dim, output_dim, output_dim_2, hidden_dim, target, num_layers, dropout, activation)

        self.num_heads = num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tanh = nn.Tanh()

        # self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, output_dim)

        # self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, hidden_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        # x = self.pe_enc(data)
        if self.target == self.TARGET_TIMESERIES:
            data = data.permute(0, 2, 1)
        x = self.linear_enc_in(data)
        x = self.encoder(x)
        x = self.linear_enc_out(x)
        # x = self.activation(x)
        x = self.tanh(x)
        if self.target == self.TARGET_TIMESERIES:
            x = x.permute(0, 2, 1)
        return x

    def decode(self, encoded):
        # x = self.pe_dec(encoded)
        if self.target == self.TARGET_TIMESERIES:
            encoded = encoded.permute(0, 2, 1)
        x = self.linear_dec_in(encoded)
        x = self.decoder(x)
        x = self.linear_dec_out(x)
        x = self.activation(x)
        if self.target == self.TARGET_TIMESERIES:
            x = x.permute(0, 2, 1)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class TransformerDoubleAutoencoder(Autoencoder):
    def __init__(self, input_dim: int, output_dim: int, output_dim_2: int, sequence_length: int, hidden_dim=256, num_layers=3, num_heads=8, dropout=0.1, activation='linear', **kwargs):
        target = Autoencoder.TARGET_BOTH
        super(TransformerDoubleAutoencoder, self).__init__(input_dim, output_dim, output_dim_2, hidden_dim, target, num_layers, dropout, activation)

        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.tanh = nn.Tanh()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder block features
        # self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, output_dim)

        # encoder block sequence
        # self.pe_enc_seq = PositionalEncoder(batch_first=True, d_model=sequence_length)
        self.linear_enc_in_seq = nn.Linear(sequence_length, hidden_dim)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=num_layers)
        self.linear_enc_out_seq = nn.Linear(hidden_dim, output_dim_2)

        # decoder block sequence
        # self.pe_dec_seq = PositionalEncoder(batch_first=True, d_model=output_dim_2)
        self.linear_dec_in_seq = nn.Linear(output_dim_2, hidden_dim)
        self.decoder_layer_seq = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder_seq = nn.TransformerEncoder(self.decoder_layer_seq, num_layers=num_layers)
        self.linear_dec_out_seq = nn.Linear(hidden_dim, sequence_length)

        # decoder block features
        # self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, hidden_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        # encoder features
        # x = self.pe_enc(data)
        x = self.linear_enc_in(data)
        x = self.encoder(x)
        x = self.linear_enc_out(x)
        x = self.tanh(x)

        # encoder sequence
        # x = self.pe_enc_seq(x.permute(0, 2, 1))
        x = self.linear_enc_in_seq(x.permute(0, 2, 1))
        x = self.encoder_seq(x)
        x = self.linear_enc_out_seq(x)
        x = self.tanh(x)
        return x.permute(0, 2, 1)

    def decode(self, encoded):
        # decoder sequence
        # x = self.pe_dec_seq(encoded.permute(0, 2, 1))
        x = self.linear_dec_in_seq(encoded.permute(0, 2, 1))
        x = self.decoder_seq(x)
        x = self.linear_dec_out_seq(x)
        x = self.activation(x)

        # decoder features
        # x = self.pe_dec(x.permute(0, 2, 1))
        x = self.linear_dec_in(x.permute(0, 2, 1))
        x = self.decoder(x)
        x = self.linear_dec_out(x)
        x = self.activation(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class TransformerFlattenAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, sequence_length, hidden_dim=1024, num_layers=3, dropout=0.1, activation='linear', **kwargs):
        super(TransformerFlattenAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout, activation)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length

        # self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_enc_out_1 = nn.Linear(sequence_length*input_dim, hidden_dim)
        self.linear_enc_out_2 = nn.Linear(hidden_dim, output_dim)

        # self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, output_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.linear_dec_out_1 = nn.Linear(output_dim, hidden_dim)
        self.linear_dec_out_2 = nn.Linear(hidden_dim, input_dim*sequence_length)

        self.tanh = nn.Sigmoid()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        # x = self.pe_enc(data)
        x = self.linear_enc_in(data).reshape(data.shape[0], self.sequence_length*self.input_dim, 1)
        x = self.encoder(x)
        x = self.linear_enc_out_1(x.permute(0, 2, 1))
        x = self.linear_enc_out_2(x)
        x = self.tanh(x)
        return x

    def decode(self, encoded):
        # x = self.pe_dec(encoded)
        x = self.linear_dec_in(encoded)
        x = self.decoder(x.permute(0, 2, 1))
        x = self.linear_dec_out_1(x.permute(0, 2, 1))
        x = self.linear_dec_out_2(x).reshape(encoded.shape[0], self.sequence_length, self.input_dim)
        x = self.tanh(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 5000,
            d_model: int = 512,
            batch_first: bool = True
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        # print(f"shape of position is {position.shape}")
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # print(f"shape of div_term is {div_term.shape}")
        pe = torch.zeros(1, max_seq_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)

        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[:, :x.size(self.x_dim)]

        return self.dropout(x)


class LSTMAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, sequence_length, hidden_dim=256, num_layers=3, dropout=0.1, activation=nn.Sigmoid(), **kwargs):
        super(LSTMAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = activation
        self.sequence_length = sequence_length

        # encoder block
        self.enc_lin_in = nn.Linear(self.input_dim, self.input_dim)
        self.enc_lstm = nn.LSTM(self.input_dim, self.output_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.enc_lin_out = nn.Linear(self.output_dim, self.output_dim)
        self.enc_dropout = nn.Dropout(self.dropout)

        # decoder block
        # decoder_block = nn.ModuleList()
        # if self.num_layers > 1:
        #     decoder_block.append(nn.Linear(self.output_dim, hidden_dim))
        #     decoder_block.append(self.activation)
        #     decoder_block.append(nn.Dropout(self.dropout))
        # if self.num_layers > 2:
        #     for _ in range(self.num_layers-2):
        #         decoder_block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        #         decoder_block.append(self.activation)
        #         decoder_block.append(nn.Dropout(self.dropout))
        # if self.num_layers == 1:
        #     decoder_block.append(nn.Linear(self.output_dim, self.input_dim*self.sequence_length))
        # else:
        #     decoder_block.append(nn.Linear(self.hidden_dim, self.input_dim*self.sequence_length))
        # decoder_block.append(self.activation)
        # self.decoder = nn.Sequential(*decoder_block)
        self.dec_lin_in = nn.Linear(self.output_dim, self.output_dim)
        self.dec_lstm = nn.LSTM(self.output_dim, self.input_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.dec_lin_out = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        # flip data along time axis
        # data = torch.flip(data, [1])
        x = self.enc_lin_in(data)
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.enc_lstm(x)[0]#.reshape(-1, self.hidden_dim//2*self.sequence_length)
        # x = self.enc_lin_out(x)
        # x = self.activation(x)
        return x

    def decode(self, encoded):
        x = self.dec_lin_in(encoded)
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.dec_lstm(x)[0]  # .reshape(-1, self.hidden_dim//2*self.sequence_length)
        # x = self.dec_lin_out(x)
        # x = self.activation(x)
        return x
        # return self.decoder(encoded)#.reshape(-1, self.sequence_length, self.input_dim)

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class LSTMDoubleAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, sequence_length, output_dim_2, hidden_dim=256, num_layers=3, dropout=0.1, activation=nn.Sigmoid(), **kwargs):
        super(LSTMDoubleAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation = activation
        self.sequence_length = sequence_length
        self.output_dim_2 = output_dim_2

        # encoder block 1
        self.enc_lin_in = nn.Linear(self.input_dim, self.input_dim)
        self.enc_lstm = nn.LSTM(self.input_dim, self.output_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.enc_lin_out = nn.Linear(self.output_dim, self.output_dim)
        self.enc_dropout = nn.Dropout(self.dropout)

        # encoder block 2
        self.enc_lin_in2 = nn.Linear(self.sequence_length, self.sequence_length)
        self.enc_lstm2 = nn.LSTM(self.sequence_length, self.output_dim_2, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.enc_lin_out2 = nn.Linear(self.output_dim_2, self.output_dim_2)

        # decoder block
        # decoder_block = nn.ModuleList()
        # if self.num_layers > 1:
        #     decoder_block.append(nn.Linear(self.output_dim, hidden_dim))
        #     decoder_block.append(self.activation)
        #     decoder_block.append(nn.Dropout(self.dropout))
        # if self.num_layers > 2:
        #     for _ in range(self.num_layers-2):
        #         decoder_block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        #         decoder_block.append(self.activation)
        #         decoder_block.append(nn.Dropout(self.dropout))
        # if self.num_layers == 1:
        #     decoder_block.append(nn.Linear(self.output_dim, self.input_dim*self.sequence_length))
        # else:
        #     decoder_block.append(nn.Linear(self.hidden_dim, self.input_dim*self.sequence_length))
        # decoder_block.append(self.activation)
        # self.decoder = nn.Sequential(*decoder_block)

        # decoder block 2
        self.dec_lin_in2 = nn.Linear(self.output_dim_2, self.output_dim_2)
        self.dec_lstm2 = nn.LSTM(self.output_dim_2, self.sequence_length, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.dec_lin_out2 = nn.Linear(self.sequence_length, self.sequence_length)

        # decoder block 1
        self.dec_lin_in = nn.Linear(self.output_dim, self.output_dim)
        self.dec_lstm = nn.LSTM(self.output_dim, self.input_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.dec_lin_out = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, data):
        return self.decode(self.encode(data))

    def encode(self, data):
        # encoder block 1
        x = self.enc_lin_in(data)
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.enc_lstm(x)[0]#.reshape(-1, self.hidden_dim//2*self.sequence_length)
        x = self.enc_lin_out(x)
        x = self.activation(x)

        # encoder block 2
        x = self.enc_lin_in2(x.permute(0, 2, 1))
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.enc_lstm2(x)[0]#.reshape(-1, self.hidden_dim//2*self.sequence_length)
        x = self.enc_lin_out2(x)
        x = self.activation(x)
        return x.permute(0, 2, 1)

    def decode(self, encoded):
        # decoder block 2
        x = self.dec_lin_in2(encoded.permute(0, 2, 1))
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.dec_lstm2(x)[0]#.reshape(-1, self.hidden_dim//2*self.sequence_length)
        x = self.dec_lin_out2(x)
        x = self.activation(x)

        # decoder block 1
        x = self.dec_lin_in(x.permute(0, 2, 1))
        x = self.activation(x)
        x = self.enc_dropout(x)
        x = self.dec_lstm(x)[0]  # .reshape(-1, self.hidden_dim//2*self.sequence_length)
        x = self.dec_lin_out(x)
        x = self.activation(x)
        return x
        # return self.decoder(encoded)#.reshape(-1, self.sequence_length, self.input_dim)

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class LSTMTransformerAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3, dropout=0.1, **kwargs):
        super(LSTMTransformerAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tanh = nn.Tanh()

        self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.enc_lstm = nn.LSTM(input_dim, output_dim, num_layers=1, dropout=dropout, batch_first=True)
        self.linear_enc_out = nn.Linear(output_dim, output_dim)

        self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, output_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=5, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.dec_lstm = nn.LSTM(output_dim, input_dim, num_layers=1, dropout=dropout, batch_first=True)
        self.linear_dec_out = nn.Linear(input_dim, input_dim)


    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        x = self.pe_enc(data)
        x = self.linear_enc_in(x)
        x = self.encoder(x)
        x = self.enc_lstm(x)[0]
        x = self.linear_enc_out(x)
        x = self.tanh(x)
        return x

    def decode(self, encoded):
        x = self.pe_dec(encoded)
        x = self.linear_dec_in(x)
        x = self.decoder(encoded)
        x = self.dec_lstm(x)[0]
        x = self.linear_dec_out(x)
        x = self.tanh(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


def train_model(model, data, optimizer, criterion, batch_size=32):
    model.train()
    total_loss = 0
    for batch in data:
        optimizer.zero_grad()
        # inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
        # inputs = filter(inputs.detach().cpu().numpy(), win_len=random.randint(29, 50), dtype=torch.Tensor)
        inputs = batch.float()
        outputs = model(inputs.to(model.device))
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)


def test_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
            inputs = batch.float()
            outputs = model(inputs.to(model.device))
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(num_epochs, model, train_data, test_data, optimizer, criterion, configuration: Optional[dict] = None):
    try:
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_data, optimizer, criterion)
            test_loss = test_model(model, test_data, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.8f}, test_loss={test_loss:.8f}")
        return train_losses, test_losses, model
    except KeyboardInterrupt:
        # save model at KeyboardInterrupt
        print("keyboard interrupt detected.")
        if configuration is not None:
            print("Configuration found.")
            configuration["model"]["state_dict"] = model.state_dict()  # update model's state dict
            save(configuration, configuration["general"]["default_save_path"])


def save(configuration, path):
    torch.save(configuration, path)
    print("Saved model and configuration to " + path)
