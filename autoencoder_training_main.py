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
from datetime import datetime
import warnings

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
    
    # create directory 'trained_models' if not exists
    if not os.path.exists('trained_ae'):
        os.makedirs('trained_ae')
        print('Directory "../trained_ae" created to store checkpoints and final model.')
    
    if default_args['load_checkpoint'] and default_args['checkpoint'] != '':
        # check if checkpoint exists and otherwise take trained_models/checkpoint.pt
        if not os.path.exists(default_args['checkpoint']):
            print(f"Checkpoint {default_args['checkpoint']} does not exist. Checkpoint is set to 'trained_models/checkpoint.pt'.")
            default_args['checkpoint'] = os.path.join('trained_ae', 'checkpoint.pt')
        print(f"Resuming training from checkpoint {default_args['checkpoint']}.")
    
    # User inputs
    opt = {
        'data': default_args['data'],
        'checkpoint': default_args['checkpoint'],
        'save_name': default_args['save_name'],
        'target': default_args['target'],
        'sample_interval': default_args['sample_interval'],
        'kw_channel': default_args['kw_channel'],
        'channels_out': default_args['channels_out'],
        'time_out': default_args['time_out'],
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
        'ddp_backend': "nccl",  #default_args['ddp_backend'],
        'norm_data': True,
        'std_data': False,
        'diff_data': False,
        'kw_time': default_args['kw_time'],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),
        'history': None,
        'trained_epochs': 0,
        'seed': default_args['seed'],
    }
    
    # set a seed for reproducibility if desired
    if opt['seed'] is not None:
        np.random.seed(opt['seed'])                       
        torch.manual_seed(opt['seed'])                    
        torch.cuda.manual_seed(opt['seed'])               
        torch.cuda.manual_seed_all(opt['seed'])           
        torch.backends.cudnn.deterministic = True
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Load, process, and split data
    # ----------------------------------------------------------------------------------------------------------------------
    
    data = Dataloader(path=opt['data'],
                      kw_channel=opt['kw_channel'], kw_time=opt['kw_time'],
                      norm_data=opt['norm_data'], std_data=opt['std_data'], diff_data=opt['diff_data'],)
    dataset = data.get_data()
    
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
    opt['channels_in'] = opt['n_channels']
    opt['time_in'] = opt['sequence_length']

    # Split dataset and convert to pytorch dataloader class
    test_dataset, train_dataset = split_data(dataset, opt['train_ratio'])
    test_dataloader = DataLoader(test_dataset, batch_size=opt['batch_size'], shuffle=True, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], shuffle=True, pin_memory=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Initiate and train autoencoder
    # ------------------------------------------------------------------------------------------------------------------

    # Initiate autoencoder
    model_dict = None
    if default_args['load_checkpoint'] and os.path.isfile(opt['checkpoint']):
        model_dict = torch.load(opt['checkpoint'])

        target_old = opt['target']
        channels_out_old = opt['channels_out']
        time_out_old = opt['time_out']

        opt['target'] = model_dict['configuration']['target']
        opt['channels_out'] = model_dict['configuration']['channels_out']
        opt['time_out'] = model_dict['configuration']['time_out']
        
        # Report changes to user
        print(f"Loading model {opt['checkpoint']}.\n\nInhereting the following parameters:")
        print("parameter:\t\told value -> new value")
        print(f"target:\t\t\t{target_old} -> {opt['target']}")
        print(f"channels_out:\t{channels_out_old} -> {opt['channels_out']}")
        print(f"time_out:\t{time_out_old} -> {opt['time_out']}")
        print('-----------------------------------\n')

    elif default_args['load_checkpoint'] and not os.path.isfile(opt['checkpoint']):
        raise FileNotFoundError(f"Checkpoint file {opt['checkpoint']} not found.")
    
    # Add parameters for tracking
    opt['input_dim'] = opt['n_channels'] if opt['target'] in ['channels', 'full'] else opt['sequence_length']
    opt['output_dim'] = opt['channels_out'] if opt['target'] in ['channels', 'full'] else opt['time_out']
    opt['output_dim_2'] = opt['sequence_length'] if opt['target'] in ['channels'] else opt['n_channels']
    
    if opt['target'] == 'channels':
        model = TransformerAutoencoder(input_dim=opt['input_dim'],
                                       output_dim=opt['output_dim'],
                                       output_dim_2=opt['output_dim_2'],
                                       target=TransformerAutoencoder.TARGET_CHANNELS,
                                       hidden_dim=opt['hidden_dim'],
                                       num_layers=opt['num_layers'],
                                       num_heads=opt['num_heads'],
                                       activation=opt['activation']).to(opt['device'])
    elif opt['target'] == 'time':
        model = TransformerAutoencoder(input_dim=opt['input_dim'],
                                       output_dim=opt['output_dim'],
                                       output_dim_2=opt['output_dim_2'],
                                       target=TransformerAutoencoder.TARGET_TIMESERIES,
                                       hidden_dim=opt['hidden_dim'],
                                       num_layers=opt['num_layers'],
                                       num_heads=opt['num_heads'],
                                       activation=opt['activation']).to(opt['device'])
    elif opt['target'] == 'full':
        model_1 = TransformerDoubleAutoencoder(channels_in=opt['channels_in'],
                                             time_in=opt['time_in'],
                                             channels_out=opt['channels_out'],
                                             time_out=opt['time_out'],
                                             hidden_dim=opt['hidden_dim'],
                                             num_layers=opt['num_layers'],
                                             num_heads=opt['num_heads'],
                                             activation=opt['activation'],
                                             training_level=1).to(opt['device'])
        
        model_2 = TransformerDoubleAutoencoder(channels_in=opt['channels_in'],
                                             time_in=opt['time_in'],
                                             channels_out=opt['channels_out'],
                                             time_out=opt['time_out'],
                                             hidden_dim=opt['hidden_dim'],
                                             num_layers=opt['num_layers'],
                                             num_heads=opt['num_heads'],
                                             activation=opt['activation'],
                                             training_level=2).to(opt['device'])

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

    training_levels = 2 if opt['target'] == 'full' else 1

    opt['training_levels'] = training_levels
    
    if opt['ddp']:
        warnings.warn(f""" The default autoencoder is a small model and DDP training adds a lot of overhead when transferring data to GPUs. 
                As such, it might be useful to test each GPU and CPU training and see what works best for your use case. 
                Although DDP training will result in better performance than CPU with the same number of training epochs,
                you can achieve this same performance quicker by adding epochs with CPU training.""", stacklevel=3)
        for training_level in range(1,training_levels+1):
            if training_levels == 2 and training_level == 1:
                print('Training the first level of the autoencoder...')
                model = model_1
            elif training_levels == 2 and training_level == 2:
                print('Training the second level of the autoencoder...')
                model = model_2
            trainer = AEDDPTrainer(model, opt)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['checkpoint'])
            mp.spawn(run, args=(opt['world_size'], find_free_port(), opt['ddp_backend'], trainer, opt),
                    nprocs=opt['world_size'], join=True)
            
            if training_levels == 2 and training_level == 1:
                model_1 = trainer.model
                model_2.model_1 = model_1
                model_2.model_1.eval()

            elif training_levels == 2 and training_level == 2:
                model_2 = trainer.model
    else:
        for training_level in range(1,training_levels+1):
            opt['training_level'] = training_level
            
            if training_levels == 2 and training_level == 1:
                print('Training the first level of the autoencoder...')
                model = model_1
            elif training_levels == 2 and training_level == 2:
                print('Training the second level of the autoencoder...')
                model = model_2
            trainer = AETrainer(model, opt)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['checkpoint'])
            samples = trainer.training(train_dataloader, test_dataloader)

            if training_levels == 2 and training_level == 1:
                model_1 = trainer.model
                model_2.model_1 = model_1
                model_2.model_1.eval()

            elif training_levels == 2 and training_level == 2:
                model_2 = trainer.model

            model = trainer.model

        print("Training finished.")

        # ----------------------------------------------------------------------------------------------------------------------
        # Save autoencoder
        # ----------------------------------------------------------------------------------------------------------------------

        # Save model
        path = 'trained_ae'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if opt['save_name'] != '':
            # check if .pt extension is already included in the save_name
            if not opt['save_name'].endswith('.pt'):
                opt['save_name'] += '.pt'
            filename = opt['save_name']
        else:
            filename = f'ae_{trainer.epochs}ep_' + timestamp + '.pt'

        opt['save_name'] = os.path.join(path, filename)
        trainer.save_checkpoint(opt['save_name'], update_history=True, samples=samples)
        print(f"Model and configuration saved in {opt['save_name']}")

if __name__ == "__main__":
    main()
