# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from nn_architecture.models import GANAE, train
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

    #Split dataset
    test_dataset, train_dataset = split_data(dataset)

    #Determine input_dim, output_dim, and seq_length
    input_dim = dataset.shape[1]-num_conditions
    output_dim = 10
    seq_length = dataset.shape[-1]
    
    #Initiate autoencoder
    model = GANAE(input_dim, output_dim, seq_length)
    
    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    test_dataset = test_dataset[:,1:,:].permute(0,2,1) #For now, just using 1 electrode
    #test_dataset = test_dataset[:,None,:]
    train_dataset = train_dataset[:,1:,:].permute(0,2,1) #For now, just using 1 electrode
    #train_dataset = train_dataset[:,None,:]
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)    

    train_loss, test_loss, model = train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)

    #Save model
    #INSERT HERE pickle...

    #Functionality
    # my_new_latent = model.encode(my_new_data)
    #my_new_decoded = model.decode(my_new_latent)

# 