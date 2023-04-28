import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=5, dim_feedforward=hidden_dim,
        #                                                 dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.linear_enc = nn.Linear(input_dim, output_dim)

        self.linear_dec = nn.Linear(output_dim, input_dim)



    def forward(self, data):
        x = self.encoder(data)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        # x = self.decoder(data, x)
        return x

    def encode(self, data):
        x = self.encoder(data)
        x = self.linear_enc(x)
        return x

    def decode(self, encoded):
        x = self.linear_dec(encoded)
        # x = self.decoder(target, x)
        return x

    def save(self, path):
        path = 'trained_ae'
        file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
        # torch.save(save, os.path.join(path, file))

def train_model(model, dataloader, optimizer, criterion):
    model.train()
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
        for batch in dataloader:
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
