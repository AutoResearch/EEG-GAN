import os
import sys
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from helpers.trainer import GANTrainer
from helpers.get_master import find_free_port
from helpers.ddp_training import run, GANDDPTrainer
from nn_architecture.models import TransformerGenerator, TransformerDiscriminator, FFGenerator, FFDiscriminator, TTSGenerator, TTSDiscriminator, DecoderGenerator, EncoderDiscriminator
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader
from helpers.initialize_gan import gan_architectures, gan_types, init_gan
from helpers import system_inputs

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""


def main():
    """Main function of the training process. 
    For input help use the command 'python gan_training_main.py help' in the terminal."""
    
    # TODO: Regarding Github Issue #62 GAN Training script does not close everything properly (aka Memory Leakage after crashed training):
    # Could not reproduce the memory leakage issue. May be an Oscar or DDP specific issue.
    # Two approaches:
    # 1. added a try/except clause at DDP Training level to destroy the process group if an exception occurs
    # 2. added a torch.cuda.empty_cache() call before each training run to clear the cache
    # First try the 1st approach. If it does not work, try the 2nd approach.
    # Here's the 2nd approach:
    # Try to fix the issue by emptying the cache before each training run.
    # torch.cuda.empty_cache()
    
    # create directory 'trained_models' if not exists
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
        print('Directory "../trained_models" created to store checkpoints and final model.')
    
    default_args = system_inputs.parse_arguments(sys.argv, file='gan_training_main.py')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp']
    ddp_backend = "nccl" #default_args['ddp_backend']
    checkpoint = default_args['checkpoint']

    # Data configuration
    diff_data = False  # Differentiate data
    std_data = False  # Standardize data
    norm_data = True  # Normalize data

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    if default_args['checkpoint'] != '':
        # check if checkpoint exists and otherwise take trained_models/checkpoint.pt
        if not os.path.exists(default_args['checkpoint']):
            print(f"Checkpoint {default_args['checkpoint']} does not exist. Checkpoint is set to 'trained_models/checkpoint.pt'.")
            default_args['checkpoint'] = os.path.join('trained_models', 'checkpoint.pt')
            checkpoint = default_args['checkpoint']
        print(f'Resuming training from checkpoint {checkpoint}.')

    # GAN configuration
    opt = {
        'gan_type': default_args['type'],
        'n_epochs': default_args['n_epochs'],
        'input_sequence_length': default_args['input_sequence_length'],
        # 'seq_len_generated': default_args['seq_len_generated'],
        'checkpoint': default_args['checkpoint'],
        'data': default_args['data'],
        'autoencoder': default_args['autoencoder'],
        'batch_size': default_args['batch_size'],
        'discriminator_lr': default_args['discriminator_lr'],
        'generator_lr': default_args['generator_lr'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['kw_conditions']) if default_args['kw_conditions'][0] != '' else 0,
        'patch_size': default_args['patch_size'],
        'kw_time': default_args['kw_time'],
        'kw_conditions': default_args['kw_conditions'],
        'sequence_length': -1,
        'hidden_dim': default_args['hidden_dim'],  # Dimension of hidden layers in discriminator and generator
        'num_layers': default_args['num_layers'],
        'activation': default_args['activation'] if default_args['autoencoder'] is None else "tanh",
        'latent_dim': 128,  # Dimension of the latent space
        'critic_iterations': 5,  # number of iterations of the critic per generator iteration for Wasserstein GAN
        'lambda_gp': 10,  # Gradient penalty lambda for Wasserstein GAN-GP
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu"), 
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count(),  # number of processes for distributed training
        # 'multichannel': default_args['multichannel'],
        'kw_channel': default_args['kw_channel'],
        'norm_data': norm_data,
        'std_data': std_data,
        'diff_data': diff_data,
        'lr_scheduler': default_args['lr_scheduler'],
        'scheduler_warmup': default_args['scheduler_warmup'],
        'scheduler_target': default_args['scheduler_target'],
        'seed': default_args['seed'],
        'save_name': default_args['save_name'],
    }
    
    # set a seed for reproducibility if desired
    if opt['seed'] is not None:
        np.random.seed(opt['seed'])                       
        torch.manual_seed(opt['seed'])                    
        torch.cuda.manual_seed(opt['seed'])               
        torch.cuda.manual_seed_all(opt['seed'])           
        torch.backends.cudnn.deterministic = True  
    
    # if autoencoder is used, take its activation function as the activation function for the generator
    # print warning that the activation function is overwritten with the autoencoder activation function
    if default_args['autoencoder'] != '':
        print(f"Warning: Since an autoencoder is used, the specified activation function {default_args['activation']} of the GAN is overwritten with the autoencoder encoding activation function 'nn.Tanh()' to ensure stability.")
    
    # Load dataset as tensor
    dataloader = Dataloader(default_args['data'],
                            kw_time=default_args['kw_time'],
                            kw_conditions=default_args['kw_conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data,
                            kw_channel=default_args['kw_channel'])
    dataset = dataloader.get_data()

    opt['channel_names'] = dataloader.channels
    opt['n_channels'] = dataset.shape[-1]
    opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    if opt['input_sequence_length'] == -1:
        opt['input_sequence_length'] = opt['sequence_length']
    opt['n_samples'] = dataset.shape[0]

    ae_dict = torch.load(opt['autoencoder'], map_location=torch.device('cpu')) if opt['autoencoder'] != '' else []
    if opt['gan_type'] == 'tts' and ae_dict and (ae_dict['configuration']['target'] == 'full' or ae_dict['configuration']['target'] == 'time') and ae_dict['configuration']['time_out'] % opt['patch_size']!= 0:
        warnings.warn(
            f"Sequence length ({ae_dict['configuration']['timeseries_out']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
            f"The sequence length is padded with zeros to fit the condition.")
        padding = 0
        while (ae_dict['configuration']['timeseries_out'] + padding) % default_args['patch_size'] != 0:
            padding += 1

        padding = torch.zeros((dataset.shape[0], padding, dataset.shape[-1]))
        dataset = torch.cat((dataset, padding), dim=1)
        opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    elif opt['gan_type'] == 'tts' and opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(
            f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
            f"The sequence length is padded with zeros to fit the condition.")
        padding = 0
        while (opt['sequence_length'] + padding) % default_args['patch_size'] != 0:
            padding += 1
        padding = torch.zeros((dataset.shape[0], padding, dataset.shape[-1]))
        dataset = torch.cat((dataset, padding), dim=1)
        opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    else:
        padding = torch.zeros((dataset.shape[0], 0, dataset.shape[-1]))

    opt['latent_dim_in'] = opt['latent_dim'] + opt['n_conditions'] + opt['n_channels'] if opt['input_sequence_length'] > 0 else opt['latent_dim'] + opt['n_conditions']
    opt['channel_in_disc'] = opt['n_channels'] + opt['n_conditions']
    opt['sequence_length_generated'] = opt['sequence_length'] - opt['input_sequence_length'] if opt['input_sequence_length'] != opt['sequence_length'] else opt['sequence_length']
    opt['padding'] = padding.shape[1]    
    
    # --------------------------------------------------------------------------------
    # Initialize generator, discriminator and trainer
    # --------------------------------------------------------------------------------
    
    generator, discriminator = init_gan(**opt)
    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    # GAN-Training
    print('\n-----------------------------------------')
    print("Training GAN...")
    print('-----------------------------------------\n')
    if ddp:
        trainer = GANDDPTrainer(generator, discriminator, opt)
        if default_args['checkpoint'] != '':
            trainer.load_checkpoint(default_args['checkpoint'])
        mp.spawn(run,
                 args=(opt['world_size'], find_free_port(), ddp_backend, trainer, opt),
                 nprocs=opt['world_size'], join=True)
        
        print("GAN training finished.")
        
    else:
        trainer = GANTrainer(generator, discriminator, opt)
        if default_args['checkpoint'] != '':
            trainer.load_checkpoint(default_args['checkpoint'])
        dataset = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=True, pin_memory=True)
        gen_samples = trainer.training(dataset)

        # save final models, optimizer states, generated samples, losses and configuration as final result
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if opt['save_name'] != '':
            # check if .pt extension is already included in the save_name
            if not opt['save_name'].endswith('.pt'):
                opt['save_name'] += '.pt'
            filename = opt['save_name']
        else:
            filename = f'gan_{trainer.epochs}ep_' + timestamp + '.pt'
        path_checkpoint = os.path.join(path, filename)
        trainer.save_checkpoint(path_checkpoint=path_checkpoint, samples=gen_samples, update_history=True)
        
        generator = trainer.generator
        discriminator = trainer.discriminator

        print("GAN training finished.")
        
        return generator, discriminator, opt, gen_samples


if __name__ == '__main__':
    main()
