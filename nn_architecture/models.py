import torch
from matplotlib import pyplot as plt
import torchaudio.functional as taf
from torch import nn
from scipy import signal
import numpy as np
from nn_architecture.ttsgan_components import *


# insert here all different kinds of generators and discriminators

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
        self.lstm1 = nn.LSTM(latent_dim+1, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
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
        self.conv2 = nn.Conv1d(hidden_size, int(hidden_size/2), kernel_size=(4,), bias=False)
        self.conv3 = nn.Conv1d(int(hidden_size/2), variables_out, kernel_size=(4,), bias=False)
        self.conv_out = nn.Conv1d(variables_out, variables_out, kernel_size=(1,), bias=True)
        # self.linear = nn.Linear(int(hidden_size/2), 1)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(int(hidden_size/2))
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
        y = self.sigmoid(y)*2
        return y


class RCDiscriminator(nn.Module):
    """Recurrent convolutional discriminator for conditional GAN"""
    def __init__(self, hidden_size=128, latent_dim=16):
        super(RCDiscriminator, self).__init__()

        self.lstm_in = nn.LSTM(1, latent_dim, num_layers=3)

        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=(4,))
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=(4,))
        self.conv3 = nn.Conv1d(hidden_size, int(hidden_size/2), kernel_size=(4,))
        self.conv_out = nn.Conv1d(int(hidden_size/2), 1, kernel_size=(4,))

        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm3 = nn.BatchNorm1d(int(hidden_size/2))
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


class TtsGenerator(nn.Module):
    """Transformer generator. Source: https://arxiv.org/abs/2202.02691"""
    def __init__(self, seq_length=600, patch_size=15, channels=1, num_classes=9, latent_dim=16, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(TtsGenerator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_length
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
            depth=self.depth,
            emb_size=self.embed_dim,
            drop_p=self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)

        return output


class TtsDiscriminator(nn.Sequential):
    """Transformer discriminator. Source: https://arxiv.org/abs/2202.02691"""
    def __init__(self,
                 in_channels=1,
                 patch_size=15,
                 emb_size=50,
                 seq_length=600,
                 depth=3,
                 n_classes=1,
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

        self.n_classes = n_classes


class TtsClassifier(nn.Sequential):
    """Transformer discriminator. Source: https://arxiv.org/abs/2202.02691"""
    def __init__(self,
                 in_channels=1,
                 patch_size=15,
                 emb_size=50,
                 seq_length=600,
                 depth=3,
                 n_classes=1,
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassifierHead(emb_size, n_classes, **kwargs)
        )


class TtsGeneratorFiltered(TtsGenerator):

    def __init__(self, seq_length=600, patch_size=15, channels=1, num_classes=9, latent_dim=16, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(TtsGeneratorFiltered, self).__init__(seq_length=seq_length,
                                                   patch_size=patch_size,
                                                   channels=channels,
                                                   num_classes=num_classes,
                                                   latent_dim=latent_dim,
                                                   embed_dim=embed_dim,
                                                   depth=depth,
                                                   num_heads=num_heads,
                                                   forward_drop_rate=forward_drop_rate,
                                                   attn_drop_rate=attn_drop_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        gen_imgs = super().forward(z)
        # outputs need to be scaled between -1 and 1 for bandpass_biquad filter
        gen_imgs = self.tanh(gen_imgs)
        # Filtering is happening here
        output = self.filter(gen_imgs)
        # rescale it back to 0 and 1 (normalized data)
        output = self.sigmoid(output)
        return output

    @staticmethod
    def filter(z, scale=False):
        """Filter the generated images to remove the noise. The last dimension of z carries the signal."""
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)
        if scale:
            # scale z between -1 and 1
            if z.max() <= 1 and z.min() >= 0:
                z = z * 2 - 1
            elif z.max() > 0 and z.min() < 0:
                z = z / torch.max(torch.abs(z))
            elif z.max() > 0 and z.min() >= 0:
                z = (z - z.mean()) / z.abs().max()
        return taf.bandpass_biquad(z, 512, 10)
