import math

import torch
from torch import nn, Tensor

from nn_architecture.ae_networks import Autoencoder

# insert here all different kinds of generators and discriminators
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256, num_layers=2, dropout=.1, **kwargs):
        """
        :param latent_dim: latent dimension
        :param output_dim: output dimension
        :param hidden_dim: hidden dimension
        :param num_layers: number of layers
        :param dropout: dropout rate

        """

        super(Generator, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.act_out = nn.ReLU()

        modulelist = nn.ModuleList()
        modulelist.append(nn.Linear(latent_dim, hidden_dim))
        modulelist.append(self.act_out)
        modulelist.append(nn.Dropout(dropout))
        for _ in range(num_layers):
            modulelist.append(nn.Linear(hidden_dim, hidden_dim))
            modulelist.append(self.act_out)
            modulelist.append(nn.Dropout(dropout))
        modulelist.append(nn.Linear(hidden_dim, output_dim))
        modulelist.append(self.act_out)

        self.block = nn.Sequential(*modulelist)

    def forward(self, z):
        return self.block(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=.1, **kwargs):
        """
        :param input_dim: input dimension
        :param hidden_dim: hidden dimension
        :param num_layers: number of layers
        :param dropout: dropout rate
        """
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.act_out = nn.ReLU()

        modulelist = nn.ModuleList()
        modulelist.append(nn.Linear(input_dim, hidden_dim))
        modulelist.append(self.act_out)
        modulelist.append(nn.Dropout(dropout))
        for _ in range(num_layers):
            modulelist.append(nn.Linear(hidden_dim, hidden_dim))
            modulelist.append(self.act_out)
            modulelist.append(nn.Dropout(dropout))
        modulelist.append(nn.Linear(hidden_dim, 1))

        self.block = nn.Sequential(*modulelist)

    def forward(self, x):
        return self.block(x)


class AutoencoderGenerator(Generator):
    """Autoencoder generator"""

    def __init__(self, latent_dim, autoencoder: Autoencoder, **kwargs):
        """
        :param autoencoder: Autoencoder model; Decoder takes in array and decodes into multidimensional array of shape (batch, sequence_length, channels)
        """
        self.output_dim = autoencoder.output_dim
        # check if output_dim_2 is attribute of autoencoder
        if hasattr(autoencoder, 'output_dim_2'):
            self.output_dim_2 = autoencoder.output_dim_2
        else:
            self.output_dim_2 = 1
        super(AutoencoderGenerator, self).__init__(latent_dim, self.output_dim*self.output_dim_2, **kwargs)
        self.autoencoder = autoencoder
        self.decode = True

    def forward(self, z):
        """
        :param z: input array of shape (batch, latent_dim)
        :return: output array of shape (batch, sequence_length, channels)
        """
        x = super(AutoencoderGenerator, self).forward(z)
        if self.decode:
            x = self.autoencoder.decode(x.reshape(-1, self.output_dim_2, self.output_dim//self.output_dim_2))
        return x

    def decode_output(self, mode=True):
        self.decode = mode


class AutoencoderDiscriminator(Discriminator):
    """Autoencoder discriminator"""

    def __init__(self, input_dim, autoencoder: Autoencoder, **kwargs):
        """
        :param autoencoder: Autoencoder model; Encoder takes in multidimensional array of shape (batch, sequence_length, channels) and encodes into array
        """
        self.output_dim = autoencoder.output_dim
        self.output_dim_2 = 1 if not hasattr(autoencoder, 'output_dim_2') else autoencoder.output_dim_2
        input_dim = input_dim - autoencoder.input_dim + self.output_dim*self.output_dim_2
        super(AutoencoderDiscriminator, self).__init__(input_dim, **kwargs)
        self.autoencoder = autoencoder
        self.encode = True

    def forward(self, z):
        """
        :param z: input array of shape (batch, sequence_length, channels + conditions)
        :return: output array of shape (batch, 1)
        """
        if self.encode:
            x = self.autoencoder.encode(z[:, :, :self.autoencoder.input_dim])
            # flatten x
            x = x.reshape(-1, 1, x.shape[-2]*x.shape[-1])
            conditions = z[:, 0, self.autoencoder.input_dim:]
            if conditions.dim() < x.dim():
                conditions = conditions.unsqueeze(1)
            x = self.block(torch.concat((x, conditions), dim=-1))
        else:
            x = self.block(z)
        return x

    def encode_input(self, mode=True):
        self.encode = mode


class CondLstmDiscriminator(nn.Module):
    """Conditional LSTM Discriminator"""

    def __init__(self, hidden_size=128, num_layers=1):
        super(CondLstmDiscriminator, self).__init__()

        self.lstm1 = nn.LSTM(2, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, data, labels):
        """
            Dimensions before processing:
            data: (batch_size, sequence_length)
            labels: (?batch_size?, sequence_length, 1)

            Dimensions after processing:
            data: (batch_size, sequence_length, 1 (channels))
            labels: (batch_size, sequence_length, 1)
        """

        if labels is not None:
            # Concatenate label and image to produce input
            d_in = data.view(data.shape[0], data.shape[1], 1)
            labels = labels.repeat(1, d_in.shape[1]).unsqueeze(-1)
            y = torch.concat((d_in, labels), 2)
        else:
            # check for correct dimensions of data
            if len(data.shape) < 3:
                raise ValueError("Data must have 3 dimensions (Batch, Sequence, Channels)"
                                 "Got {} dimensions.".format(len(data.shape)))
            if data.shape[2] != 2:
                raise ValueError("Data must have 2 channels. "
                                 "The first channel is the data, the second channel is the label")
            y = data

        y = self.lstm1(y)[0][:, -1]
        y = self.linear(y)

        return y


class CondLstmGenerator(nn.Module):
    """Conditional LSTM generator"""

    def __init__(self, hidden_size=128, latent_dim=10, num_layers=1):
        super(CondLstmGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.lstm1 = nn.LSTM(latent_dim + 1, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, 1)
        self.act_function = nn.Tanh()

    def forward(self, latent_var, labels):
        """
        Dimensions before processing:
        latent_var: (batch_size, latent_dim)
        labels: (batch_size, channels)

        Dimensions after processing:
        latent_var: (batch_size, 1 (channels), latent_dim + len(labels) (sequence length))
        """

        # Data processing: Concatenate latent variable and labels
        # Concatenate latent_var and labels_encoded to a tensor T with a channel-depth of 2
        labels = labels.unsqueeze(-1).repeat(1, latent_var.shape[1], 1)
        y = torch.concat((labels, latent_var), dim=-1)

        y = self.lstm1(y)[0]
        y = self.linear(y).squeeze(-1)
        y = self.act_function(y)

        return y


class CnnGenerator(nn.Module):
    """Convolutional generator"""

    def __init__(self, hidden_size=128, latent_dim=16, variables_out=7):
        super(CnnGenerator, self).__init__()

        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=(4,), bias=False)
        self.conv2 = nn.Conv1d(hidden_size, int(hidden_size / 2), kernel_size=(4,), bias=False)
        self.conv3 = nn.Conv1d(int(hidden_size / 2), variables_out, kernel_size=(4,), bias=False)
        self.conv_out = nn.Conv1d(variables_out, variables_out, kernel_size=(1,), bias=True)
        # self.linear = nn.Linear(int(hidden_size/2), 1)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(int(hidden_size / 2))
        self.batchnorm4 = nn.BatchNorm1d(variables_out)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=(variables_out,))

    def forward(self, latent_var, labels=None):
        """
            Dimensions before processing:
            data: (batch_size, sequence_length)
            labels: (?batch_size?, sequence_length, 1)

            Dimensions after processing:
            data: (batch_size, sequence_length, 1 (channels))
            labels: (batch_size, sequence_length, 1)
        """

        # Data processing: Concatenate latent variable and labels
        latent_var = latent_var.unsqueeze(1)

        y = self.relu(self.batchnorm2(self.conv1(latent_var)))
        y = self.relu(self.batchnorm3(self.conv2(y)))
        y = self.relu(self.batchnorm4(self.conv3(y)))
        y = self.maxpool(self.conv_out(y)).squeeze(-1)
        y = self.sigmoid(y) * 2
        return y


class RCDiscriminator(nn.Module):
    """Recurrent convolutional discriminator for conditional GAN"""

    def __init__(self, hidden_size=128, latent_dim=16):
        super(RCDiscriminator, self).__init__()

        self.lstm_in = nn.LSTM(1, latent_dim, num_layers=3)

        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=(4,))
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=(4,))
        self.conv3 = nn.Conv1d(hidden_size, int(hidden_size / 2), kernel_size=(4,))
        self.conv_out = nn.Conv1d(int(hidden_size / 2), 1, kernel_size=(4,))

        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(int(hidden_size / 2))
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=(5,))

    def forward(self, data, labels):
        """
            Dimensions before processing:
            data: (batch_size, sequence_length)
            labels: (?batch_size?, sequence_length, 1)

            Dimensions after processing:
            data: (batch_size, sequence_length, 1 (channels))
            labels: (batch_size, sequence_length, 1)
        """

        # Concatenate label and image to produce input
        # d_in = torch.concat((data, labels), 1)
        # d_in = d_in.unsqueeze(-1)
        d_in = data.unsqueeze(-1)
        y = self.relu(self.lstm_in(d_in)[0][:, -1, :])

        # Concatenate label and extracted features
        y = torch.concat((y, labels), dim=1)
        y = y.unsqueeze(1)

        y = self.relu(self.batchnorm1(self.conv1(y)))
        y = self.relu(self.batchnorm2(self.conv2(y)))
        y = self.relu(self.batchnorm3(self.conv3(y)))
        y = self.maxpool(self.conv_out(y)).squeeze(-1).squeeze(-1)

        return y


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = x + self.pe[0, :x.size(self.x_dim)]

        return self.dropout(x)


class TransformerGenerator(nn.Module):
    def __init__(self, latent_dim, channels, seq_len, hidden_dim=256, num_layers=2, num_heads=8, dropout=.1,
                 encoder=None, decoder=None, **kwargs):
        super(TransformerGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.pe = PositionalEncoder(batch_first=True, d_model=latent_dim)
        self.linear_enc_in = nn.Linear(latent_dim, hidden_dim)
        # self.linear_enc_in = nn.LSTM(latent_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, channels * seq_len)
        self.act_out = nn.Sigmoid()

        # self.deconv = nn.Sequential(
        #     nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        # )

        # TODO: Put in autoencoder
        # encoder needs as input dim n_channels
        # decoder needs as output dim n_channels
        # self.linear_enc_in and self.pe need as input dim embedding_dim of the autoencoder

        # self.encoder = encoder if encoder is not None else nn.Identity()
        # for param in self.encoder.parameters():
        #    param.requires_grad = False
        # self.decoder = decoder if decoder is not None else nn.Identity()
        # for param in self.decoder.parameters():
        #    param.requires_grad = False

    def forward(self, data):
        # x = self.pe(data)
        x = self.linear_enc_in(data)  # [0] --> only for lstm
        x = self.encoder(x)
        x = self.act_out(self.linear_enc_out(x)[:, -1]).reshape(-1, self.seq_len, self.channels)
        # x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # output = self.deconv(x.permute(0, 3, 1, 2))
        # output = output.view(-1, self.channels, H, W)
        # x = self.mask(x, data[:, :, self.latent_dim - self.channels:].diff(dim=1))
        # x = self.tanh(x)
        # x = self.decoder(x)
        return x

    def mask(self, data, data_ref, mask=0):
        # mask predictions if ALL preceding values (axis=sequence) were 'mask'
        # return indices to mask
        mask_index = (data_ref.sum(dim=1) == mask).unsqueeze(1).repeat(1, data.shape[1], 1)
        data[mask_index] = mask
        return data

class TransformerDiscriminator(nn.Module):
    def __init__(self, channels, n_classes=1, hidden_dim=256, num_layers=2, num_heads=8, dropout=.1, **kwargs):
        super(TransformerDiscriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.channels = channels
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.pe = PositionalEncoder(batch_first=True, d_model=channels)
        self.linear_enc_in = nn.Linear(channels, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, n_classes)
        self.tanh = nn.Tanh()

        # self.decoder = decoder if decoder is not None else nn.Identity()
        # for param in self.decoder.parameters():
        #    param.requires_grad = False

    def forward(self, data):
        # x = self.pe(data)
        x = self.linear_enc_in(data)
        x = self.encoder(x)
        x = self.linear_enc_out(x)[:, -1]  # .reshape(-1, self.channels)
        # x = self.mask(x, data[:,:,self.latent_dim-self.channels:].diff(dim=1))
        # x = self.tanh(x)
        # x = self.decoder(x)
        return x
  
'''
# ----------------------------------------------------------------------------------------------------------------------
# Autoencoders
# ----------------------------------------------------------------------------------------------------------------------

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3, dropout=0.1, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        #self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(input_dim, output_dim)

        #self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, output_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec_out = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        #x = self.pe_enc(data)
        #x = self.linear_enc_in(x)
        x = self.linear_enc_in(data)
        x = self.encoder(x)
        x = self.linear_enc_out(x)
        x = self.tanh(x)
        return x

    def decode(self, encoded):
        #x = self.pe_dec(encoded)
        #x = self.linear_dec_in(x)
        x = self.linear_dec_in(encoded)
        x = self.decoder(x)
        x = self.linear_dec_out(x)
        x = self.tanh(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))
        
class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=3, dropout=0.1, **kwargs):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder block of linear layers constructed in a loop and passed to a sequential container
        encoder_block = []
        encoder_block.append(nn.Linear(input_dim, hidden_dim))
        encoder_block.append(nn.Dropout(dropout))
        encoder_block.append(self.activation)
        for i in range(num_layers):
            encoder_block.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_block.append(nn.Dropout(dropout))
            encoder_block.append(self.activation)
        encoder_block.append(nn.Linear(hidden_dim, output_dim))
        encoder_block.append(self.activation)
        self.encoder = nn.Sequential(*encoder_block)

        # decoder block of linear layers constructed in a loop and passed to a sequential container
        decoder_block = []
        decoder_block.append(nn.Linear(output_dim, hidden_dim))
        decoder_block.append(nn.Dropout(dropout))
        decoder_block.append(self.activation)
        for i in range(num_layers):
            decoder_block.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_block.append(nn.Dropout(dropout))
            decoder_block.append(self.activation)
        decoder_block.append(nn.Linear(hidden_dim, input_dim))
        decoder_block.append(self.activation)
        self.decoder = nn.Sequential(*decoder_block)

    def forward(self, x):
        encoded = self.encoder(x.to(self.device))
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, data):
        return self.encoder(data.to(self.device))

    def decode(self, encoded):
        return self.decoder(encoded)

class TransformerFlattenAutoencoder(Autoencoder):
    def __init__(self, input_dim, output_dim, sequence_length, hidden_dim=1024, num_layers=3, dropout=0.1, **kwargs):
        super(TransformerFlattenAutoencoder, self).__init__(input_dim, output_dim, hidden_dim, num_layers, dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length

        #self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_enc_out_1 = nn.Linear(sequence_length*input_dim, hidden_dim)
        self.linear_enc_out_2 = nn.Linear(hidden_dim, output_dim)

        #self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
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
        #x = self.pe_enc(data)
        #x = self.linear_enc_in(x).reshape(data.shape[0], self.sequence_length*self.input_dim, 1)
        x = self.linear_enc_in(data).reshape(data.shape[0], self.sequence_length*self.input_dim, 1)
        x = self.encoder(x)
        x = self.linear_enc_out_1(x.permute(0, 2, 1))
        x = self.linear_enc_out_2(x)
        x = self.tanh(x)
        return x

    def decode(self, encoded):
        #x = self.pe_dec(encoded)
        #x = self.linear_dec_in(x)
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
        
class TransformerDoubleAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_length, output_dim_2, hidden_dim=256, num_layers=3, dropout=0.1, **kwargs):
        super(TransformerDoubleAutoencoder, self).__init__()
        
        # parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # encoder block features
        #self.pe_enc = PositionalEncoder(batch_first=True, d_model=input_dim)
        self.linear_enc_in = nn.Linear(input_dim, input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(input_dim, output_dim)

        # encoder block sequence
        #self.pe_enc_seq = PositionalEncoder(batch_first=True, d_model=sequence_length)
        self.linear_enc_in_seq = nn.Linear(sequence_length, sequence_length)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=sequence_length, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=num_layers)
        self.linear_enc_out_seq = nn.Linear(sequence_length, output_dim_2)

        # decoder block sequence
        #self.pe_dec_seq = PositionalEncoder(batch_first=True, d_model=output_dim_2)
        self.linear_dec_in_seq = nn.Linear(output_dim_2, output_dim_2)
        self.decoder_layer_seq = nn.TransformerEncoderLayer(d_model=output_dim_2, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder_seq = nn.TransformerEncoder(self.decoder_layer_seq, num_layers=num_layers)
        self.linear_dec_out_seq = nn.Linear(output_dim_2, sequence_length)

        # decoder block features
        #self.pe_dec = PositionalEncoder(batch_first=True, d_model=output_dim)
        self.linear_dec_in = nn.Linear(output_dim, output_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=2, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec_out = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        # encoder features
        #x = self.pe_enc(data)
        #x = self.linear_enc_in(x)
        x = self.linear_enc_in(data)
        x = self.encoder(x)
        x = self.linear_enc_out(x)
        x = self.tanh(x)

        # encoder sequence
        #x = self.pe_enc_seq(x.permute(0, 2, 1))
        #x = self.linear_enc_in_seq(x)
        x = self.linear_enc_in_seq(x.permute(0, 2, 1))
        x = self.encoder_seq(x)
        x = self.linear_enc_out_seq(x)
        x = self.tanh(x)
        return x.permute(0, 2, 1)

    def decode(self, encoded):
        # decoder sequence
        #x = self.pe_dec_seq(encoded.permute(0, 2, 1))
        #x = self.linear_dec_in_seq(x)
        x = self.linear_dec_in_seq(encoded.permute(0, 2, 1))
        x = self.decoder_seq(x)
        x = self.linear_dec_out_seq(x)
        x = self.tanh(x)

        # decoder features
        #x = self.pe_dec(x.permute(0, 2, 1))
        #x = self.linear_dec_in(x)
        x = self.linear_dec_in(x.permute(0, 2, 1))
        x = self.decoder(x)
        x = self.linear_dec_out(x)
        x = self.tanh(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))
         
def train_model(model, dataloader, optimizer, criterion):
    model.train() #Sets it into training mode
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch.float()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        batch = dataloader.dataset[np.random.randint(0, len(dataloader), dataloader.batch_size)]
        inputs = batch.float()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion, configuration: Optional[dict] = None):
    try:
        train_losses = []
        test_losses = []
        trigger = True
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_dataloader, optimizer, criterion)
            test_loss = test_model(model, test_dataloader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            model.config['trained_epochs'][-1] += 1
            print(f"Epoch {epoch + 1}/{num_epochs} (Model Total: {str(sum(model.config['trained_epochs']))}): train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
            trigger = save_checkpoint(model, epoch, trigger, 100)
        return train_losses, test_losses, model
    except KeyboardInterrupt:
        print("keyboard interrupt detected.")
        return train_losses, test_losses, model

def save_checkpoint(model, epoch, trigger, criterion = 100):
    if (epoch+1) % criterion == 0:
        model_dict = dict(state_dict = model.state_dict(), config = model.config)
        
        # toggle between checkpoint files to avoid corrupted file during training
        if trigger:
            save(model_dict, 'checkpoint_01.pth', verbose=False)
            trigger = False
        else:
            save(model_dict, 'checkpoint_02.pth', verbose=False)
            trigger = True
            
    return trigger
            
def save(model, file, path = 'trained_ae', verbose = True):
    torch.save(model, os.path.join(path, file))
    if verbose:
        print("Saved model and configuration to " + os.path.join(path, file))
'''