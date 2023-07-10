# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from nn_architecture.models import GANAE, train, save
from helpers.dataloader import Dataloader

if __name__ == '__main__':

    #User inputs
    filename = "data/gansMultiCondition.csv"
    num_conditions = 1
    num_epochs = 100

    #Load and process data
    data = Dataloader(filename, col_label='Condition', channel_label='Electrode')
    dataset = data.get_data()
    
    #Split data function
    def split_data(dataset, test_size=.3):
        num_samples = dataset.shape[0]
        shuffle_index = np.arange(num_samples)
        np.random.shuffle(shuffle_index)
        
        cutoff_index = int(num_samples*test_size)
        test = dataset[shuffle_index[0:cutoff_index]]
        train = dataset[shuffle_index[cutoff_index:]]
        
        return test, train

    #Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)    

    #Determine input_dim, output_dim, and seq_length
    input_dim = dataset.shape[1]-num_conditions
    output_dim = 10 #The time-series size in the encoded layer (TODO: Turn this into a parameter)
    seq_length = dataset.shape[-1]
    
    #Initiate autoencoder
    model = GANAE(input_dim, output_dim, seq_length)
    
    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_loss, test_loss, model = train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)

    #Save model
    save_name = filename.split('/')[-1].split('.csv')[0]
    save(model, f'trained_ae/ae_{save_name}.pth')

    #Functionality
    # my_new_latent = model.encode(my_new_data)
    #my_new_decoded = model.decode(my_new_latent)

# 