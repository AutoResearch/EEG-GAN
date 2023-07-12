# train an autoencoder with attention mechanism for multivariate time series
import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nn_architecture.models import TransformerAutoencoder, TransformerDoubleAutoencoder, train, save
from helpers.dataloader import Dataloader
from helpers import system_inputs

def main():

    #default_args = system_inputs.parse_arguments(sys.argv, file='autoencoder_training_main.py')

    #User inputs
    '''
    file = default_args['file'] #"data/gansMultiCondition.csv"
    num_epochs = default_args['num_epochs'] #4000
    conditions = default_args['conditions'] #['Condition']
    channel_label = default_args['channel_label'] #['Electrode']
    num_conditions = len(conditions)    
    timeseries_out = default_args['timeseries_out'] #10
    channel_out = default_args['channels_out'] #2
    batch_size = default_args['batch_size'] #32
    '''
    
    file = 'data/gansMultiCondition.csv'
    path_checkpoint = 'trained_ae/ae_gansMultiCondition_both_nepochs4.pth'
    target = 'full' #'channels', 'timeseries' (not implemented yet), or 'both'
    n_epochs = 4
    conditions = 'Condition'
    channel_label = 'Electrode'
    num_conditions = 1   
    timeseries_out = 10
    channel_outs = 6
    batch_size = 32

    #Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Scale function
    def scale(dataset):
        x_min, x_max = dataset.min(), dataset.max()
        return (dataset-x_min)/(x_max-x_min)
        
    #Load and process data
    data = Dataloader(file, col_label=conditions, channel_label=channel_label)
    dataset = data.get_data()
    dataset = dataset[:,num_conditions:,:].to(device) #Remove labels
    dataset = scale(dataset)
    
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
    input_dim = dataset.shape[1] #-num_conditions
    seq_length = dataset.shape[-1]
    
    #Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    #Initiate autoencoder
    if path_checkpoint:
        model = torch.load(path_checkpoint)
    elif target == 'channels':
        model = TransformerAutoencoder(input_dim=seq_length, output_dim=channel_outs).to(device)
    elif target == 'timeseries':
        raise ValueError("Timeseries encoding target is not yet implemented")
    elif target == 'full':
        model = TransformerDoubleAutoencoder(input_dim=seq_length, output_dim=channel_outs, sequence_length=input_dim, output_dim_2=timeseries_out).to(device) 
    else:
        raise ValueError(f"Encode target '{target}' not recognized, options are 'channels', 'timeseries', or 'full'.")
    
    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_loss, test_loss, model = train(n_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)

    #Save model
    if model: 
        fn = file.split('/')[-1].split('.csv')[0]
        save(model, f"ae__{fn}__{target}_nepochs{str(n_epochs)}_{str(time.time()).split('.')[0]}.pth")
    
if __name__ == "__main__":
    main()