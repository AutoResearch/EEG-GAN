import os
import random
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

# from utils.get_filter import moving_average as filter


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden_dec=256, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear_enc = nn.Linear(input_dim, output_dim)

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.linear_dec = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.tanh(self.encode(data.to(self.device)))
        x = self.tanh(self.decode(x))
        return x

    def encode(self, data):
        x = self.encoder(data.to(self.device))
        x = self.linear_enc(x)
        return x

    def decode(self, encoded):
        x = self.decoder(encoded)
        x = self.linear_dec(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class TransformerAutoencoder_v0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden_dec=256, **kwargs):
        super(TransformerAutoencoder_v0, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear_enc = nn.Linear(input_dim, output_dim)

        self.linear_dec = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.encode(data.to(self.device))
        x = self.decode(x)
        return x

    def encode(self, data):
        x = self.encoder(data.to(self.device))
        x = self.linear_enc(x)
        return x

    def decode(self, encoded):
        x = self.linear_dec(encoded)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden_dec=256, **kwargs):
        super(LSTMAutoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        # self.linear_enc = nn.Linear(input_dim, output_dim)

        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        # self.linear_dec = nn.Linear(output_dim, input_dim)

        self.tanh = nn.Tanh()

    def forward(self, data):
        x = self.tanh(self.encode(data.to(self.device))[0])
        x = self.tanh(self.decode(x)[0])
        return x

    def encode(self, data):
        x = self.encoder(data.to(self.device))
        # x = self.linear_enc(x)
        return x

    def decode(self, encoded):
        x = self.decoder(encoded)
        # x = self.linear_dec(x)
        return x

    def save(self, path):
        path = '../trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))


def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch.float()
        # inputs = filter(inputs.detach().cpu().numpy(), win_len=random.randint(29, 50), dtype=torch.Tensor)
        outputs = model(inputs.to(model.device))
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def test_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.float()
            outputs = model(inputs.to(model.device))
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
