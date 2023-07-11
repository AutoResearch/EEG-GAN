# train an autoencoder with attention mechanism for multivariate time series
import os.path

import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from nn_architecture.models import TransformerDoubleAutoencoder, train, save
from helpers.dataloader import Dataloader
import time

if __name__ == '__main__':

    #User inputs
    filename = "data/gansMultiCondition.csv"
    num_conditions = 1
    num_epochs = 500

    #Load and process data
    data = Dataloader(filename, col_label='Condition', channel_label='Electrode')
    dataset = data.get_data()
    dataset = dataset[:,1:,:] #Remove labels
    
    #DEBUG: Pairing down to one electrode to see if it trains better. If this is still here, remove it.
    #dataset = dataset[:,:,0].unsqueeze(2)
    
    #Split data function
    def split_data(dataset, test_size=.3):
        num_samples = dataset.shape[0]
        shuffle_index = np.arange(num_samples)
        np.random.shuffle(shuffle_index)
        
        cutoff_index = int(num_samples*test_size)
        test = dataset[shuffle_index[0:cutoff_index]]
        train = dataset[shuffle_index[cutoff_index:]]
        
        return test, train

    #Determine input_dim, output_dim, and seq_length
    input_dim = dataset.shape[1]#-num_conditions
    output_dim = 10 #The time-series size in the encoded layer (TODO: Turn this into a parameter)
    seq_length = dataset.shape[-1]
    output_dim_2 = 6
    
    '''
    #No longer needed:
    #Adjust inputs and outputs to include labels and ensure they are even sizes
    input_dim = input_dim + 1 #opt['n_conditions']
    output_dim = output_dim + 1 #opt['n_conditions']
    # make sure latent_dim_in is even; constraint of positional encoding in TransformerGenerator
    
    if input_dim % 2 != 0:
        dataset = torch.cat((dataset, torch.zeros(dataset.shape[0],1,dataset.shape[2])), dim=1) #Pad data with zeros
        input_dim += 1
        
    if output_dim % 2 != 0:
        output_dim +=1
    '''
        
    #Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)    
        
    #Initiate autoencoder
    model = TransformerDoubleAutoencoder(input_dim=seq_length, output_dim=output_dim, sequence_length=input_dim, output_dim_2=output_dim_2)
    
    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_loss, test_loss, model = train(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)

    #Save model
    save_name = filename.split('/')[-1].split('.csv')[0]
    current_time = str(time.time()).split('.')[0]
    save(model, f'trained_ae/ae_{save_name}_{current_time}.pth')

    #Functionality
    # my_new_latent = model.encode(my_new_data)
    #my_new_decoded = model.decode(my_new_latent)

# 