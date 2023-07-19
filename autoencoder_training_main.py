# train an autoencoder with attention mechanism for multivariate time series
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nn_architecture.models import TransformerAutoencoder, TransformerFlattenAutoencoder, TransformerDoubleAutoencoder, train, save
from helpers.dataloader import Dataloader
from helpers import system_inputs

def main():


    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters
    # ----------------------------------------------------------------------------------------------------------------------

    default_args = system_inputs.parse_arguments(sys.argv, file='autoencoder_training_main.py')
    print('-----------------------------------\n')

    #User inputs
    file = default_args['file']
    path_checkpoint = default_args['path_checkpoint']
    save_name = default_args['save_name']
    target = default_args['target']
    conditions = default_args['conditions']
    channel_label = default_args['channel_label']
    channels_out = default_args['channels_out']
    timeseries_out = default_args['timeseries_out']
    n_epochs = default_args['n_epochs']
    batch_size = default_args['batch_size']
    
    num_conditions = len(conditions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Scale function
    def scale(dataset):
        x_min, x_max = dataset.min(), dataset.max()
        return (dataset-x_min)/(x_max-x_min)
        
    # ----------------------------------------------------------------------------------------------------------------------
    # Load, process, and split data
    # ----------------------------------------------------------------------------------------------------------------------

    data = Dataloader(file, col_label=conditions, channel_label=channel_label)
    dataset = data.get_data()
    dataset = dataset[:,num_conditions:,:].to(device) #Remove labels
    dataset = scale(dataset)
    
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
    input_dim = dataset.shape[-1]
    seq_length = dataset.shape[1]
    
    #Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    # ----------------------------------------------------------------------------------------------------------------------
    # Initiate and train autoencoder
    # ----------------------------------------------------------------------------------------------------------------------

    #Initiate autoencoder
    if path_checkpoint:
        model = torch.load(path_checkpoint)
        target = model.config['target']
        channels_out = model.config['channels_out']
        timeseries_out = model.config['timeseries_out']
        
        #Report changes to user
        print(f"Loading model {path_checkpoint}. \n\nInhereting the following parameters:")
        print(f"Target: {target}")
        if (target == 'channels') | (target == 'full'):
            print(f"channels_out: {channels_out}")
        if (target == 'timeseries') | (target == 'full'):
            print(f"timeseries_out: {timeseries_out}")
            print('-----------------------------------\n')
    elif target == 'channels':
        model = TransformerAutoencoder(input_dim=input_dim, output_dim=channels_out).to(device)
    elif target == 'timeseries':
        raise ValueError("Timeseries encoding target is not yet implemented")
    elif target == 'full':
        model = TransformerFlattenAutoencoder(input_dim=input_dim, sequence_length=seq_length, output_dim=channels_out).to(device) 
    else:
        raise ValueError(f"Encode target '{target}' not recognized, options are 'channels', 'timeseries', or 'full'.")

    #Populate model configuration
    config = {
        "file" : file,
        "path_checkpoint" : path_checkpoint,
        "save_name" : save_name,
        "target" : target,
        "conditions" : conditions,
        "channel_label" : channel_label,
        "channels_out" : channels_out,
        "timeseries_out" : timeseries_out,
        "n_epochs" : [n_epochs],
        "batch_size" : [batch_size],
        "trained_epochs": 0,
    }
    
    if path_checkpoint:
        model_nepochs = copy.deepcopy(model.config['n_epochs'])
        model_nepochs.append(n_epochs)
        config['n_epochs'] = model_nepochs
        
        model_batch = copy.deepcopy(model.config['batch_size'])
        model_batch.append(batch_size)
        config['batch_size'] = model_batch  

        config['trained_epochs'] = model.config['trained_epochs']
  
    model.config = config

    #Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_loss, test_loss, model = train(n_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)

    # ----------------------------------------------------------------------------------------------------------------------
    # Save autoencoder
    # ----------------------------------------------------------------------------------------------------------------------

    #Save model
    if model: 
        if save_name == None:
            fn = file.split('/')[-1].split('.csv')[0]
            save(model, f"ae_{fn}_{str(time.time()).split('.')[0]}.pth")
        else:
            save(model, save_name)
        
if __name__ == "__main__":
    main()