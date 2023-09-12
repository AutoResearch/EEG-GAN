# train an autoencoder with attention mechanism for multivariate time series
import sys
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from nn_architecture.ae_networks import TransformerAutoencoder, TransformerFlattenAutoencoder, TransformerDoubleAutoencoder, train, save
from helpers.dataloader import Dataloader
from helpers import system_inputs
from helpers.trainer import AETrainer
from helpers.ddp_training import AEDDPTrainer, run
from helpers.get_master import find_free_port


def main():

    # ------------------------------------------------------------------------------------------------------------------
    # Configure training parameters
    # ------------------------------------------------------------------------------------------------------------------

    default_args = system_inputs.parse_arguments(sys.argv, file='autoencoder_training_main.py')
    print('-----------------------------------------\n')

    # User inputs
    opt = {
        'path_dataset': default_args['path_dataset'],
        'path_checkpoint': default_args['path_checkpoint'],
        'save_name': default_args['save_name'],
        'target': default_args['target'],
        'sample_interval': default_args['sample_interval'],
        # 'conditions': default_args['conditions'],
        'channel_label': default_args['channel_label'],
        'channels_out': default_args['channels_out'],
        'timeseries_out': default_args['timeseries_out'],
        'n_epochs': default_args['n_epochs'],
        'batch_size': default_args['batch_size'],
        'train_ratio': default_args['train_ratio'],
        'learning_rate': default_args['learning_rate'],
        'hidden_dim': default_args['hidden_dim'],
        'num_heads': default_args['num_heads'],
        'num_layers': default_args['num_layers'],
        'activation': default_args['activation'],
        'learning_rate': default_args['learning_rate'],
        'num_heads': default_args['num_heads'],
        'num_layers': default_args['num_layers'],
        'ddp': default_args['ddp'],
        'ddp_backend': default_args['ddp_backend'],
        # 'n_conditions': len(default_args['conditions']) if default_args['conditions'][0] != '' else 0,
        'norm_data': True,
        'std_data': False,
        'diff_data': False,
        'kw_timestep': default_args['kw_timestep'],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),
        'history': None,
        'trained_epochs': 0
    }

    # ----------------------------------------------------------------------------------------------------------------------
    # Load, process, and split data
    # ----------------------------------------------------------------------------------------------------------------------

    # Scale function -> Not necessary; already in dataloader -> param: norm_data=True
    # def scale(dataset):
    #     x_min, x_max = dataset.min(), dataset.max()
    #     return (dataset-x_min)/(x_max-x_min)
    
    data = Dataloader(path=opt['path_dataset'],
                      channel_label=opt['channel_label'], kw_timestep=opt['kw_timestep'],
                      norm_data=opt['norm_data'], std_data=opt['std_data'], diff_data=opt['diff_data'],)
    dataset = data.get_data()
    # dataset = dataset[:, opt['n_conditions']:, :].to(opt['device']) #Remove labels
    # dataset = scale(dataset)
    
    # Split data function
    def split_data(dataset, train_size=.8):
        num_samples = dataset.shape[0]
        shuffle_index = np.arange(num_samples)
        np.random.shuffle(shuffle_index)
        
        cutoff_index = int(num_samples*train_size)
        train = dataset[shuffle_index[:cutoff_index]]
        test = dataset[shuffle_index[cutoff_index:]]

        return test, train

    # Determine n_channels, output_dim, and seq_length
    opt['n_channels'] = dataset.shape[-1]
    opt['sequence_length'] = dataset.shape[1]

    # Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset, opt['train_ratio'])
    test_dataloader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Initiate and train autoencoder
    # ------------------------------------------------------------------------------------------------------------------

    # Initiate autoencoder
    model_dict = None
    if default_args['load_checkpoint'] and os.path.isfile(opt['path_checkpoint']):
        model_dict = torch.load(opt['path_checkpoint'])
        # model_state = model_dict['state_dict']

        target_old = opt['target']
        channels_out_old = opt['channels_out']
        timeseries_out_old = opt['timeseries_out']

        opt['target'] = model_dict['configuration']['target']
        opt['channels_out'] = model_dict['configuration']['channels_out']
        opt['timeseries_out'] = model_dict['configuration']['timeseries_out']
        
        # Report changes to user
        print(f"Loading model {opt['path_checkpoint']}.\n\nInhereting the following parameters:")
        print("parameter:\t\told value -> new value")
        print(f"target:\t\t\t{target_old} -> {opt['target']}")
        print(f"channels_out:\t{channels_out_old} -> {opt['channels_out']}")
        print(f"timeseries_out:\t{timeseries_out_old} -> {opt['timeseries_out']}")
        print('-----------------------------------\n')
        # print(f"Target: {opt['target']}")
        # if (opt['target'] == 'channels') | (opt['target'] == 'full'):
        #     print(f"channels_out: {opt['channels_out']}")
        # if (opt['target'] == 'timeseries') | (opt['target'] == 'full'):
        #     print(f"timeseries_out: {opt['timeseries_out']}")
        #     print('-----------------------------------\n')

    elif default_args['load_checkpoint'] and not os.path.isfile(opt['path_checkpoint']):
        raise FileNotFoundError(f"Checkpoint file {opt['path_checkpoint']} not found.")
    
    # Add parameters for tracking
    opt['input_dim'] = opt['n_channels'] if opt['target'] in ['channels', 'full'] else opt['sequence_length']
    opt['output_dim'] = opt['channels_out'] if opt['target'] in ['channels', 'full'] else opt['n_channels']
    opt['output_dim_2'] = opt['sequence_length'] if opt['target'] in ['channels'] else opt['timeseries_out']
    
    if opt['target'] == 'channels':
        model = TransformerAutoencoder(input_dim=opt['n_channels'],
                                       output_dim=opt['channels_out'],
                                       output_dim_2=opt['sequence_length'],
                                       target=TransformerAutoencoder.TARGET_CHANNELS,
                                       hidden_dim=opt['hidden_dim'],
                                       num_layers=opt['num_layers'],
                                       num_heads=opt['num_heads'],).to(opt['device'])
    elif opt['target'] == 'time':
        model = TransformerAutoencoder(input_dim=opt['sequence_length'],
                                       output_dim=opt['timeseries_out'],
                                       output_dim_2=opt['n_channels'],
                                       target=TransformerAutoencoder.TARGET_TIMESERIES,
                                       hidden_dim=opt['hidden_dim'],
                                       num_layers=opt['num_layers'],
                                       num_heads=opt['num_heads'],).to(opt['device'])
    elif opt['target'] == 'full':
        model = TransformerDoubleAutoencoder(input_dim=opt['n_channels'],
                                             output_dim=opt['output_dim'],
                                             output_dim_2=opt['output_dim_2'],
                                             sequence_length=opt['sequence_length'],
                                             hidden_dim=opt['hidden_dim'],
                                             num_layers=opt['num_layers'],
                                             num_heads=opt['num_heads'],).to(opt['device'])
    else:
        raise ValueError(f"Encode target '{opt['target']}' not recognized, options are 'channels', 'time', or 'full'.")

    # Populate model configuration    
    history = {}
    for key in opt.keys():
        if (not key == 'history') | (not key == 'trained_epochs'):
            history[key] = [opt[key]]
    history['trained_epochs'] = []

    if model_dict is not None:
        # update history
        for key in history.keys():
            history[key] = model_dict['configuration']['history'][key] + history[key]

    opt['history'] = history

    if opt['ddp']:
        trainer = AEDDPTrainer(model, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        mp.spawn(run, args=(opt['world_size'], find_free_port(), opt['ddp_backend'], trainer, opt),
                 nprocs=opt['world_size'], join=True)
    else:
        trainer = AETrainer(model, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        samples = trainer.training(train_dataloader, test_dataloader)
        model = trainer.model
        print("Training finished.")

        # ----------------------------------------------------------------------------------------------------------------------
        # Save autoencoder
        # ----------------------------------------------------------------------------------------------------------------------

        # Save model
        # model_dict = dict(state_dict=model.state_dict(), config=model.config)
        if opt['save_name'] is None:
            fn = opt['path_dataset'].split('/')[-1].split('.csv')[0]
            opt['save_name'] = os.path.join("trained_ae", f"ae_{fn}_{str(time.time()).split('.')[0]}.pt")
        # save(model_dict, save_name)
    
        trainer.save_checkpoint(opt['save_name'], update_history=True, samples=samples)
        print(f"Model and configuration saved in {opt['save_name']}")

if __name__ == "__main__":
    main()
