# train an autoencoder with attention mechanism for multivariate time series
import os.path

import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from dataloader import create_dataloader
from transformer_autoencoder import TransformerAutoencoder, train, save

if __name__ == '__main__':

    # get parameters from saved model
    load_model = False

    model_dict = None
    model_name = 'ae_20230428_145919.pth'
    model_dir = 'trained_ae'
    if load_model:
        model_dict = torch.load(os.path.join(model_dir, model_name))

    data_dir = 'stock_data'
    data_file = 'stocks_sp500_2010_2020.csv'  # path to the csv file

    # configuration
    cfg = {
        "model": {
            "state_dict":   None,
            "input_dim":    None,
            "hidden_dim":   64,
            "output_dim":   20,
            "num_layers":   2,
            "dropout":      .1,
        },
        "training": {
            "lr":           1e-3,
            "epochs":       2,
        },
        "general": {
            "seq_len":      24,
            "scaler":       None,
            "training_data": os.path.join(data_dir, data_file),
            "batch_size":   32,
            "train_ratio":  .8,
            "standardize":  True,
            "differentiate": True,
            "default_save_path": os.path.join('trained_ae', 'checkpoint.pth'),
        }
    }

    if model_dict is not None:
        cfg = model_dict
        print("adapted configuration from saved file " + os.path.join(model_dir, model_name))

    # load data from csv file as DataLoader
    train_dataloader, test_dataloader, scaler = create_dataloader(**cfg["general"])
    cfg["general"]["scaler"] = scaler

    # create the model
    if cfg["model"]["input_dim"] is None:
        cfg["model"]["input_dim"] = train_dataloader.dataset.data.shape[1]
    model = TransformerAutoencoder(**cfg["model"])

    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    # train the model
    train_losses, test_losses, model = train(num_epochs=cfg["training"]["epochs"], model=model, train_dataloader=train_dataloader,
                                             test_dataloader=test_dataloader, optimizer=optimizer, criterion=criterion, configuration=cfg)

    # save model and training history under file with name model_CURRENTDATETIME.pth
    cfg["model"]["state_dict"] = model.state_dict()
    # get filename as ae_ + timestampe + pth
    filename = 'trained_ae.pth'
    save(cfg, os.path.join("trained_ae", filename))

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

    # decode a batch of sequences, rescale it with scaler and plot them
    outputs = model.decode(outputs)
    print(outputs.shape)
    # outputs = scaler.inverse_transform(outputs.detach().numpy())
    # inputs = scaler.inverse_transform(inputs.detach().numpy())
    plt.plot(outputs[0, :, 0].detach().numpy())
    plt.plot(inputs[0, :, 0].detach().numpy())
    plt.legend(['reconstructed', 'original'])
    plt.show()


