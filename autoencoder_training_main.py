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
        model_dict = torch.load(path_checkpoint)
        model_state = model_dict['state_dict']
        
        target = model_dict['config']['target']
        channels_out = model_dict['config']['channels_out']
        timeseries_out = model_dict['config']['timeseries_out']
        
        #Report changes to user
        print(f"Loading model {path_checkpoint}. \n\nInhereting the following parameters:")
        print(f"Target: {target}")
        if (target == 'channels') | (target == 'full'):
            print(f"channels_out: {channels_out}")
        if (target == 'timeseries') | (target == 'full'):
            print(f"timeseries_out: {timeseries_out}")
            print('-----------------------------------\n')
            
    if target == 'channels':
        model = TransformerAutoencoder(input_dim=input_dim, output_dim=channels_out).to(device)
    elif target == 'timeseries':
        raise ValueError("Timeseries encoding target is not yet implemented")
    elif target == 'full':
        #model = TransformerFlattenAutoencoder(input_dim=input_dim, sequence_length=seq_length, output_dim=channels_out).to(device) 
        model = TransformerDoubleAutoencoder(input_dim=input_dim, output_dim=channels_out, sequence_length=seq_length , output_dim_2=timeseries_out).to(device) 
    else:
        raise ValueError(f"Encode target '{target}' not recognized, options are 'channels', 'timeseries', or 'full'.")

    if path_checkpoint:
        model.load_state_dict(model_state)
        
    #Populate model configuration
    config = {
        "file" : [file],
        "path_checkpoint" : path_checkpoint,
        "save_name" : save_name,
        "target" : target,
        "conditions" : [conditions],
        "channel_label" : [channel_label],
        "channels_out" : channels_out,
        "timeseries_out" : timeseries_out,
        "n_epochs" : [n_epochs],
        "batch_size" : [batch_size],
        "trained_epochs": [0],
    }
    
    if path_checkpoint:
        model_file = copy.deepcopy(model_dict['config']['file'])
        model_file.append(file)
        config['file'] = model_file
        
        model_conditions = copy.deepcopy(model_dict['config']['conditions'])
        model_conditions.append(conditions)
        config['conditions'] = model_conditions
        
        model_channel_label = copy.deepcopy(model_dict['config']['channel_label'])
        model_channel_label.append(channel_label)
        config['channel_label'] = model_channel_label
        
        model_nepochs = copy.deepcopy(model_dict['config']['n_epochs'])
        model_nepochs.append(n_epochs)
        config['n_epochs'] = model_nepochs
        
        model_batch = copy.deepcopy(model_dict['config']['batch_size'])
        model_batch.append(batch_size)
        config['batch_size'] = model_batch  
        
        model_trained = copy.deepcopy(model_dict['config']['trained_epochs'])
        model_trained.append(0)
        config['trained_epochs'] = model_trained
  
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
        model_dict = dict(state_dict = model.state_dict(), config = model.config)
        if save_name == None:
            fn = file.split('/')[-1].split('.csv')[0]
            save_name = f"ae_{fn}_{str(time.time()).split('.')[0]}.pth"
        save(model_dict, save_name)
        
if __name__ == "__main__":
    main()