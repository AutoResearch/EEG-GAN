import random

import numpy as np
import torch
import torch.nn as nn


# embedding network to reduce the dimension of time-series data; not tested yet!


class Encoder(nn.Module):
    """Encoder for embedding net."""

    def __init__(self, input_size=1, hidden_size=128, embedding_dim=16, num_layers=1, seq_len=600):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embedding_dim, bias=False)

    def forward(self, data):
        """
            Dimensions before processing:
            data: (batch_size, sequence_length, sequence + conditions)
        """

        y = self.lstm(data)[0][:, -1]
        y = self.linear(y)#[:, -1]

        return y


class Decoder(nn.Module):
    """Decoder for embedding net."""

    def __init__(self, signals=1, conditions=1, hidden_size=128, embedding_dim=16, num_layers=1, seq_len=600):
        super(Decoder, self).__init__()

        self.seq_len = seq_len

        self.linear = nn.Linear(embedding_dim, hidden_size, bias=False)
        self.lstm_signal = nn.LSTM(hidden_size, signals, num_layers=num_layers, batch_first=True)
        self.linear_condition = nn.Linear(hidden_size, conditions, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        """
            Dimensions before processing:
            data: (batch_size, sequence_length)

            Dimensions after processing:
            data: (batch_size, sequence_length, latent_dim)
        """

        # make a sequence of the embedding
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        data = data.unsqueeze(1).repeat(1, self.seq_len, 1)

        y = self.linear(data)
        y_signal = self.lstm_signal(y)[0]
        y_condition = self.sigmoid(self.linear_condition(y))

        y = torch.concat((y_condition, y_signal), 2)

        return y


class Trainer:
    """Trainer for embedding net."""

    def __init__(self, encoder: Encoder, decoder: Decoder, opt):
        # training configuration

        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 1
        self.lr = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder
        self.decoder = decoder
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.criterion = nn.MSELoss()
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

    def train(self, dataset, data_start_index=1):
        """Train the embedding net on the given data."""
        self.encoder.train()
        self.decoder.train()

        # move dataset to device and get labels and data
        random.shuffle(dataset)
        data = dataset[:, data_start_index:].unsqueeze(-1).to(self.device)
        if data_start_index == 1:
            labels = dataset[:, 0].to(self.device)
        else:
            labels = dataset[:, 0:data_start_index-1].to(self.device)
        labels = labels.unsqueeze(-1).unsqueeze(-1).repeat(1, data.shape[1], 1)
        data = torch.concat((labels, data), dim=-1)

        # split data and labels into training and validation set
        train_size = int(0.8 * len(data))
        train_data, test_data = data[:train_size], data[train_size:]

        # containers
        container = []
        losses = []

        for i in range(self.epochs):
            for j in range(0, train_data.shape[0], self.batch_size):
                # get remaining batch size
                batch_size = min(self.batch_size, data.shape[0] - j)

                # get batch
                batch = train_data[j:j+batch_size]

                # encode batch
                embedding = self.encoder(batch)

                # decode batch
                reconstruction = self.decoder(embedding)

                # calculate loss
                loss = self.criterion(reconstruction, batch)

                # back-propagate
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                # test on validation set
                test_loss = self.test(test_data)

                # collect every self.sample_interval batches the last entry of batch and reconstruction
                batches_done = (i + 1) * int(j/self.batch_size)
                if batches_done % self.sample_interval == 0:
                    real_data = np.concatenate((batch[-1, 0, 0].view(-1, 1).detach().cpu().numpy(),
                                                batch[-1, :, 1].view(-1, 1).detach().cpu().numpy()))
                    reconstructed_data = np.concatenate((reconstruction[-1, 0, 0].view(-1, 1).detach().cpu().numpy(),
                                                         reconstruction[-1, :, 1].view(-1, 1).detach().cpu().numpy()))
                    container.append(real_data)
                    container.append(reconstructed_data)

                # print loss
                print(f'Epoch: [{i}/{self.epochs}], '
                      f'Batch: [{int(j/self.batch_size)}/{int(train_data.shape[0]/self.batch_size)}], '
                      f'Training loss: {loss.item()}, Test loss: {test_loss}')
                losses.append(np.array((loss.item(), test_loss)))

        return self.encoder, self.decoder, container, losses

    def test(self, data):
        """Test the embedding net on the given evaluation data."""
        self.encoder.eval()
        self.decoder.eval()

        # move data to device
        data = data.to(self.device)

        # test encoder
        embedding = self.encoder(data)
        reconstruction = self.decoder(embedding)
        loss = self.criterion(reconstruction, data)

        return loss.item()

    def encode(self, data):
        """Encode the given data."""
        self.encoder.eval()
        self.decoder.eval()

        # move data to device
        data = data.to(self.device)

        # encode data
        embedding = self.encoder(data)

        return embedding

    def decode(self, data):
        """Decode the given data."""
        self.encoder.eval()
        self.decoder.eval()

        # move data to device
        data = data.to(self.device)

        # decode data
        reconstruction = self.decoder(data)

        return reconstruction
