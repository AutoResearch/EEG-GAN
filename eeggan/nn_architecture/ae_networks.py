import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    TARGET_CHANNELS = 0
    TARGET_TIMESERIES = 1
    TARGET_BOTH = 2

    def __init__(self, input_dim: int, output_dim: int, output_dim_2: int, hidden_dim: int, target: int, num_layers=3, dropout=0.1, activation_decoder='linear', **kwargs):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_2 = output_dim_2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.target = target
        self.activation_encoder = nn.Tanh()
        if activation_decoder == 'relu':
            self.activation_decoder = nn.ReLU()
        elif activation_decoder == 'sigmoid':
            self.activation_decoder = nn.Sigmoid()
        elif activation_decoder == 'tanh':
            self.activation_decoder = nn.Tanh()
        elif activation_decoder == 'leakyrelu':
            self.activation_decoder = nn.LeakyReLU()
        elif activation_decoder == 'linear':
            self.activation_decoder = nn.Identity()
        else:
            raise ValueError(f"Activation function of type '{activation_decoder}' was not recognized.")
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
        encoder_block.append(self.activation_encoder)
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
        decoder_block.append(self.activation_decoder)
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

    def __init__(self, input_dim: int, output_dim: int, output_dim_2: int, target: int, hidden_dim=256, num_layers=3, num_heads=4, dropout=0.1, activation_decoder='linear', **kwargs):
        super(TransformerAutoencoder, self).__init__(input_dim, output_dim, output_dim_2, hidden_dim, target, num_layers, dropout, activation_decoder)

        self.num_heads = num_heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.activation_encoder(x)
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
        x = self.activation_decoder(x)
        if self.target == self.TARGET_TIMESERIES:
            x = x.permute(0, 2, 1)
        return x


class TransformerDoubleAutoencoder(Autoencoder):
    def __init__(self, channels_in: int, time_in: int, channels_out: int, time_out: int, hidden_dim=256, num_layers=3, num_heads=8, dropout=0.1, activation_decoder='linear', training_level=2, **kwargs):
        target = Autoencoder.TARGET_BOTH
        super(TransformerDoubleAutoencoder, self).__init__(channels_in, channels_out, time_out, hidden_dim, target, num_layers, dropout, activation_decoder)

        '''
        Note that this double autoencoder trains two autoencoders - the first is a timeseries autoencoder and the second is a channels autoencoder.
        Whereas the first autoencoder is doing the same as the single TransformerAutoencoder with the target=timeseries,
        the second autoencoder first encodes the data via the timeseries autoencoder and then learns to encode the channel dimension from that.

        After extensive testing, training the timeseries autoencoder first and the channels autoencoder second is much quicker and more effective than
        training the channels autoencoder first and the timeseries autoencoder second. 
        '''
        
        self.training_level = training_level
        self.sequence_length = time_in
        self.num_heads = num_heads

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Timeseries Encoder
        self.linear_enc_in_timeseries = nn.Linear(time_in, hidden_dim)
        self.encoder_layer_timeseries = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder_timeseries = nn.TransformerEncoder(self.encoder_layer_timeseries, num_layers=num_layers)
        self.linear_enc_out_timeseries = nn.Linear(hidden_dim, time_out)

        # Channel Encoder
        self.linear_enc_in_channels = nn.Linear(channels_in, hidden_dim)
        self.encoder_layer_channels = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder_channels = nn.TransformerEncoder(self.encoder_layer_channels, num_layers=num_layers)
        self.linear_enc_out_channels = nn.Linear(hidden_dim, channels_out)

        # Channel Decoder
        self.linear_dec_in_channels = nn.Linear(channels_out, hidden_dim)
        self.decoder_layer_channels = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder_channels = nn.TransformerEncoder(self.decoder_layer_channels, num_layers=num_layers)
        self.linear_dec_out_channels = nn.Linear(hidden_dim, channels_in)
        
        # Timeseries Decoder
        self.linear_dec_in_timeseries = nn.Linear(time_out, hidden_dim)
        self.decoder_layer_timeseries = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder_timeseries = nn.TransformerEncoder(self.decoder_layer_timeseries, num_layers=num_layers)
        self.linear_dec_out_timeseries = nn.Linear(hidden_dim, time_in)

    def forward(self, x):
        x = self.encode(x.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, x):
        if self.training_level == 1:

            #Encode timeseries
            x = x.permute(0, 2, 1)
            x = self.linear_enc_in_timeseries(x)
            x = self.encoder_timeseries(x)
            x = self.linear_enc_out_timeseries(x)
            x = self.activation_encoder(x)
            x = x.permute(0, 2, 1)
        
        if self.training_level == 2: 

            #Encode timeseries
            x = self.model_1.encode(x) 

            #Encode channels
            x = self.linear_enc_in_channels(x)
            x = self.encoder_channels(x)
            x = self.linear_enc_out_channels(x)
            x = self.activation_encoder(x)

        return x

    def decode(self, x):
        if self.training_level == 1:

            x = x.permute(0, 2, 1)
            x = self.linear_dec_in_timeseries(x)
            x = self.decoder_timeseries(x)
            x = self.linear_dec_out_timeseries(x)
            x = self.activation_decoder(x)
            x = x.permute(0, 2, 1)
        
        if self.training_level == 2:

            #Decode timeseries
            x = self.model_1.decode(x)

            #Decode channels
            x = self.linear_dec_in_channels(x)
            x = self.decoder_channels(x)
            x = self.linear_dec_out_channels(x)
            x = self.activation_decoder(x)

        return x