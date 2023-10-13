import math
import warnings

import torch
from torch import nn, Tensor

from nn_architecture.ae_networks import Autoencoder
from nn_architecture.tts_gan_components import Generator as TTSGenerator_Org, Discriminator as TTSDiscriminator_Org

# insert here all different kinds of generators and discriminators
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class FFGenerator(Generator):
    def __init__(self, latent_dim, channels, seq_len, hidden_dim=256, num_layers=4, dropout=.1, activation='tanh', **kwargs):
        """
        :param latent_dim: latent dimension
        :param channels: output dimension
        :param hidden_dim: hidden dimension
        :param num_layers: number of layers
        :param dropout: dropout rate

        """

        super(Generator, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout = dropout
        if activation == 'relu':
            self.act_out = nn.ReLU()
        elif activation == 'sigmoid':
            self.act_out = nn.Sigmoid()
        elif activation == 'tanh':
            self.act_out = nn.Tanh()
        elif activation == 'leakyrelu':
            self.act_out = nn.LeakyReLU()
        elif activation == 'linear':
            self.act_out = nn.Identity()
        else:
            self.act_out = nn.Identity()
            warnings.warn(
                f"Activation function of type '{activation}' was not recognized. Activation function was set to 'linear'.")

        modulelist = nn.ModuleList()
        modulelist.append(nn.Linear(latent_dim, hidden_dim))
        modulelist.append(nn.LeakyReLU(0.1))
        modulelist.append(nn.Dropout(dropout))
        for _ in range(num_layers):
            modulelist.append(nn.Linear(hidden_dim, hidden_dim))
            modulelist.append(nn.LeakyReLU(0.1))
            modulelist.append(nn.Dropout(dropout))
        modulelist.append(nn.Linear(hidden_dim, channels * seq_len))
        modulelist.append(self.act_out)

        self.block = nn.Sequential(*modulelist)

    def forward(self, z):
        return self.block(z).reshape(-1, self.seq_len, self.channels)


class FFDiscriminator(Discriminator):
    def __init__(self, channels, seq_len, hidden_dim=256, num_layers=4, dropout=.1, **kwargs):
        """
        :param channels: input dimension
        :param hidden_dim: hidden dimension
        :param num_layers: number of layers
        :param dropout: dropout rate
        """
        super(Discriminator, self).__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        modulelist = nn.ModuleList()
        modulelist.append(nn.Linear(channels * seq_len, hidden_dim))
        modulelist.append(nn.LeakyReLU(0.1))
        modulelist.append(nn.Dropout(dropout))
        for _ in range(num_layers):
            modulelist.append(nn.Linear(hidden_dim, hidden_dim))
            modulelist.append(nn.LeakyReLU(0.1))
            modulelist.append(nn.Dropout(dropout))
        modulelist.append(nn.Linear(hidden_dim, 1))

        self.block = nn.Sequential(*modulelist)

    def forward(self, x):
        return self.block(x.reshape(-1, 1, self.seq_len * self.channels))


class AutoencoderGenerator(FFGenerator):
    """Autoencoder generator"""

    def __init__(self, latent_dim, autoencoder: Autoencoder, **kwargs):
        """
        :param autoencoder: Autoencoder model; Decoder takes in array and decodes into multidimensional array of shape (batch, sequence_length, channels)
        """
        self.output_dim_1 = autoencoder.output_dim if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim_2
        self.output_dim_2 = autoencoder.output_dim_2 if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim
        super(AutoencoderGenerator, self).__init__(latent_dim, self.output_dim_1*self.output_dim_2, **kwargs)
        self.autoencoder = autoencoder
        self.decode = True
        
    def forward(self, z):
        """
        :param z: input array of shape (batch, latent_dim)
        :return: output array of shape (batch, sequence_length, channels)
        """
        x = super(AutoencoderGenerator, self).forward(z)
        if self.decode:
            x = self.autoencoder.decode(x.reshape(-1, self.output_dim_2, self.channels // self.output_dim_2))
        return x

    def decode_output(self, mode=True):
        self.decode = mode


class AutoencoderDiscriminator(FFDiscriminator):
    """Autoencoder discriminator"""

    def __init__(self, channels, autoencoder: Autoencoder, **kwargs):
        """
        :param autoencoder: Autoencoder model; Encoder takes in multidimensional array of shape (batch, sequence_length, channels) and encodes into array
        """
        n_channels = autoencoder.input_dim if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim_2
        channels = channels - n_channels + autoencoder.output_dim * autoencoder.output_dim_2
        super(AutoencoderDiscriminator, self).__init__(channels, **kwargs)
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
            max_seq_len: int = 100,
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


class TransformerGenerator(Generator):
    def __init__(self, latent_dim, channels, seq_len, hidden_dim=8, num_layers=2, num_heads=4, dropout=.1, **kwargs):
        super(TransformerGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.channels = channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.pe = PositionalEncoder(batch_first=True, d_model=latent_dim)
        self.linear_enc_in = nn.Linear(latent_dim, hidden_dim*seq_len)
        # self.linear_enc_in = nn.LSTM(latent_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=dropout,
                                                        batch_first=True,
                                                        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, channels)
        self.act_out = nn.Tanh()

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
        x = self.linear_enc_in(data).reshape(-1, self.seq_len, self.hidden_dim)  # [0] --> only for lstm
        x = self.encoder(x)
        x = self.act_out(self.linear_enc_out(x))
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


class TransformerDiscriminator(Discriminator):
    def __init__(self, channels, seq_len, n_classes=1, hidden_dim=8, num_layers=2, num_heads=4, dropout=.1, **kwargs):
        super(TransformerDiscriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.channels = channels
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.pe = PositionalEncoder(batch_first=True, d_model=channels)
        self.linear_enc_in = nn.Linear(channels, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=dropout,
                                                        batch_first=True,
                                                        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim*seq_len, n_classes)
        self.tanh = nn.Tanh()

        # self.decoder = decoder if decoder is not None else nn.Identity()
        # for param in self.decoder.parameters():
        #    param.requires_grad = False

    def forward(self, data):
        # x = self.pe(data)
        x = self.linear_enc_in(data)
        x = self.encoder(x).reshape(-1, 1, self.seq_len*self.hidden_dim)
        x = self.linear_enc_out(x)  # .reshape(-1, self.channels)
        # x = self.mask(x, data[:,:,self.latent_dim-self.channels:].diff(dim=1))
        # x = self.tanh(x)
        # x = self.decoder(x)
        return x


class TTSGenerator(TTSGenerator_Org):
    def __init__(self, seq_len=150, patch_size=15, channels=3, num_classes=9, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(TTSGenerator, self).__init__(seq_len, patch_size, channels, num_classes, latent_dim, embed_dim, depth,
                 num_heads, forward_drop_rate, attn_drop_rate)


class TTSDiscriminator(TTSDiscriminator_Org):
    def __init__(self, in_channels=3, patch_size=15, emb_size=50, seq_length=150, depth=3, n_classes=1, **kwargs):
        super(TTSDiscriminator, self).__init__(in_channels, patch_size, emb_size, seq_length, depth, n_classes, **kwargs)

class DecoderGenerator(Generator):
    """
    DecoderGenerator serves as a wrapper for a generator.
    It takes the output of the generator and passes it to a given decoder if the corresponding flag was set.
    Otherwise, it returns the output of the generator.
    """

    def __init__(self, generator: Generator, decoder: Autoencoder):
        """
        :param generator: generator model
        :param decoder: autoencoder model that has a decode method
        """
        super(DecoderGenerator, self).__init__()
        self.generator = generator
        self.decoder = decoder
        self.decode = True

        # add attributes from generator
        self.latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else None
        self.channels = generator.channels if hasattr(generator, 'channels') else None
        self.seq_len = generator.seq_len if hasattr(generator, 'seq_len') else None


    def forward(self, data):
        if self.decode:
            return self.decoder.decode(self.generator(data))
        else:
            return self.generator(data)

    def decode_output(self, mode=True):
        self.decode = mode


class EncoderDiscriminator(Discriminator):
    """
    EncoderDiscriminator serves as a wrapper for a discriminator.
    It takes the input of the discriminator and passes it to a given encoder if the corresponding flag was set.
    Otherwise, it returns the output of the discriminator.
    """

    def __init__(self, discriminator: Discriminator, encoder: Autoencoder):
        """
        :param discriminator: discriminator model
        :param encoder: autoencoder model that has an encode method
        """
        super(EncoderDiscriminator, self).__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.encode = True

        # add attributes from discriminator
        self.channels = discriminator.channels if hasattr(discriminator, 'channels') else None
        self.n_classes = discriminator.n_classes if hasattr(discriminator, 'n_classes') else None

    def forward(self, data):
        if self.encode:
            return self.encoder.encode(self.discriminator(data))
        else:
            return self.discriminator(data)

    def encode_input(self, mode=True):
        self.encode = mode


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