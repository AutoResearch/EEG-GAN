# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch

from helpers.ae_dataloader import create_dataloader
from helpers.get_filter import moving_average as filter
from nn_architecture.transformer_autoencoder import TransformerAutoencoder, save, train

if __name__ == '__main__':

    # get parameters from saved model
    load_model = False
    training = True

    model_dict = None
    model_dir = 'trained_ae'
    model_name = 'ae_kagglev1.pth'

    data_dir = 'data'
    data_file = 'gansMultiCondition.csv'  # path to the csv file

    # configuration
    cfg = {
        "model": {
            "state_dict":   None,
            "input_dim":    None,
            "hidden_dim":   512,
            "output_dim":   10,
            "num_layers":   2,
            "dropout":      .3,
        },
        "training": {
            "lr":           1e-4,
            "epochs":       10,
        },
        "general": {
            "seq_len":      50,
            "scaler":       None,
            "training_data": os.path.join(data_dir, data_file),
            "batch_size":   32,
            "train_ratio":  .8,
            "standardize":  True,
            "differentiate": False,
            "default_save_path": os.path.join('trained_ae', 'checkpoint.pt'),
        }
    }

    # load model
    if load_model:
        cfg["model"] = torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu'))["model"]
        print("adapted configuration from saved file " + os.path.join(model_dir, model_name))

    # load data from csv file as DataLoader
    cfg["general"]["training_data"] = os.path.join(data_dir, data_file)
    train_dataloader, test_dataloader, scaler = create_dataloader(**cfg["general"])
    cfg["general"]["scaler"] = scaler

    # create the model
    if cfg["model"]["input_dim"] is None:
        cfg["model"]["input_dim"] = train_dataloader.dataset.data.shape[2]
    model = TransformerAutoencoder(**cfg["model"])

    # create the optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    if training:
        # train the model
        train_losses, test_losses, model = train(num_epochs=cfg["training"]["epochs"], model=model, train_dataloader=train_dataloader,
                                                 test_dataloader=test_dataloader, optimizer=optimizer, criterion=criterion, configuration=cfg)

        # save model and training history under file with name model_CURRENTDATETIME.pth
        cfg["model"]["state_dict"] = model.state_dict()
        # get filename as ae_ + timestampe + pth
        save(cfg, os.path.join("trained_ae", model_name))

        # plot the training and test losses
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(['train', 'test'])
        plt.show()

    # encode a batch of sequences
    batch = next(iter(test_dataloader))[0]
    win_lens = np.random.randint(29, 50, size=batch.shape[-1])
    inputs = batch.float()
    inputs_filtered = torch.zeros_like(inputs)
    for i in range(batch.shape[-1]):
        inputs_filtered[:, i] = filter((inputs[:, i]-inputs[0, i]).detach().cpu().numpy(), win_len=win_lens[i], dtype=torch.Tensor)
    outputs = model.encode(inputs_filtered.to(model.device))

    # decode a batch of sequences, rescale it with scaler and plot them
    outputs = model.decode(outputs)
    # outputs = scaler.inverse_transform(outputs.detach().cpu().numpy())
    # inputs = scaler.inverse_transform(inputs.detach().cpu().numpy())
    fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
    for i in range(10):
        stock = np.random.randint(0, inputs.shape[-1])
        # out = scaler.inverse_transform(outputs[i, :].detach().cpu().numpy())
        # inp = scaler.inverse_transform(inputs[i, :].detach().cpu().numpy())
        axs[i].plot(inputs[:, stock].detach().cpu().numpy(), label='Original')
        axs[i].plot(inputs_filtered[:, stock].detach().cpu().numpy(), label='Filter')
        axs[i].plot(outputs[:, stock].detach().cpu().numpy(), label='Reconstructed')
        # axs[i, 1].plot(np.cumsum(inputs[:, stock].detach().cpu().numpy()), label='Original')
        # axs[i, 1].plot(np.cumsum(inputs_filtered[:, stock].detach().cpu().numpy()), label='Filter')
        # axs[i, 1].plot(np.cumsum(outputs[:, stock].detach().cpu().numpy()), label='Reconstructed')
    plt.legend()
    plt.show()

