import os
import sys
import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from helpers import system_inputs
from helpers.dataloader import Dataloader
from helpers.trainer import VAETrainer
from helpers.get_master import find_free_port
from helpers.ddp_training import run#, VAEDDPTrainer
from nn_architecture.vae_networks import VariationalAutoencoder

def main():
    # ------------------------------------------------------------------------------------------------------------------
    # Configure training parameters
    # ------------------------------------------------------------------------------------------------------------------

    default_args = system_inputs.parse_arguments(sys.argv, file='vae_training_main.py')
    print('-----------------------------------------\n')

    if default_args['load_checkpoint']:
        print(f'Resuming training from checkpoint {default_args["path_checkpoint"]}.')

    #User input
    opt = {
        'data': default_args['data'],
        'path_checkpoint': default_args['path_checkpoint'],
        'save_name': default_args['save_name'],
        'sample_interval': default_args['sample_interval'],
        'kw_channel': default_args['kw_channel'],
        'kw_conditions': default_args['kw_conditions'],
        'n_epochs': default_args['n_epochs'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'hidden_dim': default_args['hidden_dim'],
        'encoded_dim': default_args['encoded_dim'],
        'activation': default_args['activation'],
        'kl_alpha': default_args['kl_alpha'],
        'norm_data': True,
        'std_data': False,
        'diff_data': False,
        'kw_time': default_args['kw_time'],
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),
        'history': None,
        'trained_epochs': 0
    }

    #opt['device'] = torch.device("cuda" if torch.cuda.is_available() and opt['ddp'] else "cpu")
    opt['device'] = torch.device("cpu")

    # raise warning if no normalization and standardization is used at the same time
    if opt['std_data'] and opt['norm_data']:
        raise Warning("Standardization and normalization are used at the same time.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Load, process, and split data
    # ----------------------------------------------------------------------------------------------------------------------
    data = Dataloader(path=opt['data'],
                      kw_channel=opt['kw_channel'], 
                      kw_conditions=opt['kw_conditions'],
                      kw_time=opt['kw_time'],
                      norm_data=opt['norm_data'], 
                      std_data=opt['std_data'], 
                      diff_data=opt['diff_data'])
    dataset = data.get_data()

    opt['input_dim'] = (dataset.shape[1] - len(opt['kw_conditions'])) * dataset.shape[-1]

    # ------------------------------------------------------------------------------------------------------------------
    # Load VAE checkpoint and populate configuration
    # ------------------------------------------------------------------------------------------------------------------
    
    # Load VAE
    model_dict = None
    if default_args['load_checkpoint'] and os.path.isfile(opt['path_checkpoint']):
        model_dict = torch.load(opt['path_checkpoint'])
    elif default_args['load_checkpoint'] and not os.path.isfile(opt['path_checkpoint']):
        raise FileNotFoundError(f"Checkpoint file {opt['path_checkpoint']} not found.")
    
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

    # ------------------------------------------------------------------------------------------------------------------
    # Initiate VAE
    # ------------------------------------------------------------------------------------------------------------------

    model = VariationalAutoencoder(input_dim=opt['input_dim'], 
                                   hidden_dim=opt['hidden_dim'], 
                                   encoded_dim=opt['encoded_dim'], 
                                   activation=opt['activation'],
                                   device=opt['device']).to(opt['device'])
    
    print('Variational autoencoder initialized')

    # ------------------------------------------------------------------------------------------------------------------
    # Train VAE
    # ------------------------------------------------------------------------------------------------------------------
    
    # VAE-Training
    print('\n-----------------------------------------')
    print("Training VAE...")
    print('-----------------------------------------\n')
    
    trainer = VAETrainer(model, opt)
    if default_args['load_checkpoint']:
        trainer.load_checkpoint(default_args['path_checkpoint'])
    dataset = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=True)
    gen_samples = trainer.training(dataset)

    # save final models, optimizer states, generated samples, losses and configuration as final result
    if not opt['save_name']:
        path = 'trained_vae'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'vae_{trainer.epochs}ep_' + timestamp + '.pt'
        save_filename = os.path.join(path, filename)
    else:
        save_filename = opt['save_name']
    trainer.save_checkpoint(path_checkpoint=save_filename, samples=gen_samples, update_history=True)

    print(f"Checkpoint saved to {default_args['path_checkpoint']}.")

    model = trainer.model

    print("VAE training finished.")
    print(f"Model states and generated samples saved to file {save_filename}.")

    return model, opt, gen_samples

if __name__ == '__main__':
    main()