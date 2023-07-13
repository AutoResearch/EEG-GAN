import os
import torch
import torchaudio.functional as taf
from torch import nn
import pandas as pd
from nn_architecture.ttsgan_components import *
from typing import Optional

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


class TransformerGenerator2(nn.Module):
    def __init__(self, latent_dim, channels, seq_len, hidden_dim=256, num_layers=2, num_heads=8, dropout=.1,  **kwargs):
        super(TransformerGenerator2, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pe = PositionalEncoder(batch_first=True, d_model=latent_dim)
        # self.linear_enc_in = nn.Linear(latent_dim, hidden_dim)
        self.linear_enc_in = nn.LSTM(latent_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear_enc_out = nn.Linear(hidden_dim, channels * seq_len)
        self.tanh = nn.Tanh()

        # TODO: Put it in autoencoder
        # self.decoder = decoder if decoder is not None else nn.Identity()
        # for param in self.decoder.parameters():
        #    param.requires_grad = False

    def forward(self, data):
        x = self.pe(data.to(self.device))
        x = self.linear_enc_in(x)[0]
        x = self.encoder(x)
        x = self.linear_enc_out(x)[:, -1].reshape(-1, self.seq_len, self.channels)
        x = self.mask(x, data[:, :, self.latent_dim - self.channels:].diff(dim=1))
        x = self.tanh(x)
        # x = self.decoder(x)
        return x  # .unsqueeze(2).permute(0, 3, 2, 1)

    def mask(self, data, data_ref, mask=0):
        # mask predictions if ALL preceding values (axis=sequence) were 'mask'
        # return indices to mask
        mask_index = (data_ref.sum(dim=1) == mask).unsqueeze(1).repeat(1, data.shape[1], 1)
        data[mask_index] = mask
        return data
    
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
        for batch in dataloader.dataset[np.random.randint(0, len(dataloader), dataloader.batch_size)].unsqueeze(0):
            inputs = batch.float()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion, configuration: Optional[dict] = None):
    try:
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_dataloader, optimizer, criterion)
            test_loss = test_model(model, test_dataloader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
            save_checkpoint(model, epoch, 100)
        return train_losses, test_losses, model
    except KeyboardInterrupt:
        # save model at KeyboardInterrupt
        print("keyboard interrupt detected.")
        return train_losses, test_losses, model

def save_checkpoint(model, epoch, criterion = 100):
    if (epoch+1) % criterion == 0:
        if not os.path.isfile('trained_ae/checkpoint_01.pth'):
            torch.save(model,'trained_ae/checkpoint_01.pth')
            if os.path.isfile('trained_ae/checkpoint_02.pth'):
                os.remove('trained_ae/checkpoint_02.pth') 
        else: 
            torch.save(model,'trained_ae/checkpoint_02.pth')
            os.remove('trained_ae/checkpoint_01.pth') 
            
def save(model, file, path = 'trained_ae'):
    torch.save(model, os.path.join(path, file))
    print("Saved model and configuration to " + os.path.join(path, file))
    if os.path.isfile('trained_ae/checkpoint_01.pth'):
        os.remove('trained_ae/checkpoint_01.pth')
    if os.path.isfile('trained_ae/checkpoint_02.pth'):
        os.remove('trained_ae/checkpoint_02.pth')