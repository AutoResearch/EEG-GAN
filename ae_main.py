# train an autoencoder with attention mechanism for multivariate time series
import os.path

import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from dataloader import create_dataloader
from transformer_autoencoder import TransformerAutoencoder, train

if __name__ == '__main__':
    # config data loader
    data_dir = 'stock_data'
    data_file = 'stocks_sp500_2010_2020.csv'  # path to the csv file
    seq_len = 24  # sequence length
    batch_size = 32  # batch size
    train_ratio = 0.8  # train-test split ratio
    standardize = True  # standardize the data
    differentiate = True  # differentiate the data

    # config model
    input_dim = None  # input dimension
    hidden_dim = 64  # hidden dimension
    output_dim = 50  # output dimension
    num_layers = 2  # number of layers
    dropout = 0.1  # dropout rate

    # config training
    lr = 1e-3  # learning rate
    num_epochs = 10  # number of epochs

    # load data from csv file as DataLoader
    train_dataloader, test_dataloader = create_dataloader(os.path.join(data_dir, data_file),
                                                          seq_len, batch_size, train_ratio,
                                                          standardize=standardize, differentiate=differentiate)

    # create the model
    if input_dim is None:
        input_dim = train_dataloader.dataset.data.shape[1]
    model = TransformerAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                   num_layers=num_layers, dropout=dropout)

    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # train the model
    train_losses, test_losses, main = train(num_epochs=num_epochs, model=model, train_dataloader=train_dataloader,
                                            test_dataloader=test_dataloader, optimizer=optimizer, criterion=criterion)

    # plot the training and test losses
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.show()

    # encode a batch of sequences
    batch = next(iter(test_dataloader))
    inputs = batch.float()
    outputs = model.encode(inputs)
    print(outputs.shape)

    # decode a batch of sequences
    outputs = model.decode(outputs)
    print(outputs.shape)
    plt.plot(outputs[0, :, 0].detach().numpy())
    plt.plot(inputs[0, :, 0].detach().numpy())
    plt.legend(['reconstructed', 'original'])
    plt.show()

    # save model and training history under file with name model_CURRENTDATETIME.pth
    path = 'trained_ae'
    file = f'ae_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pth'
    torch.save(model.state_dict(), os.path.join(path, file))
