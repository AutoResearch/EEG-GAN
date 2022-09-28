import sys
from datetime import datetime
import torch

import torch.multiprocessing as mp

from get_master import find_free_port
from ddp_training import run, DDPTrainer as Trainer
from models import TtsDiscriminator, TtsGeneratorFiltered as Generator#, TtsGeneratorFiltered
from dataloader import Dataloader

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""

# TODO: update the training process and the GANs so the distinction between TTS-GAN and other GANs is not necessary

if __name__ == '__main__':

    # Get system arguments
    n_epochs, sequence_length, seq_len_generated, trained_gan, train_gan, \
    windows_slices, patch_size, batch_size, learning_rate, sample_interval, n_conditions, path_dataset = \
        None, None, None, None, None, None, None, None, None, None, None, None
    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')
    for arg in sys.argv:
        if '=' in arg:
            kw = arg.split('=')
            if kw[0] == 'n_epochs':
                print(f'Number of epochs: {kw[1]}')
                n_epochs = int(kw[1])
            elif kw[0] == 'sequence_length':
                print(f'Total sequence length: {kw[1]}')
                sequence_length = int(kw[1])
            elif kw[0] == 'seq_len_generated':
                print(f'Sequence length to generate: {kw[1]}')
                seq_len_generated = int(kw[1])
            elif kw[0] == 'trained_gan':
                print(f'Use pre-trained GAN: {kw[1]}')
                trained_gan = kw[1] == 'True'
            elif kw[0] == 'train_gan':
                print(f'Train GAN: {kw[1]}')
                train_gan = kw[1] == 'True'
            elif kw[0] == 'windows_slices':
                print(f'Use window slices: {kw[1]}')
                windows_slices = kw[1] == 'True'
            elif kw[0] == 'patch_size':
                print(f'Patch size of transformer: {kw[1]}')
                patch_size = int(kw[1])
            elif kw[0] == 'batch_size':
                print(f'Batch size: {kw[1]}')
                batch_size = int(kw[1])
            elif kw[0] == 'learning_rate':
                print(f'Learning rate: {kw[1]}')
                learning_rate = float(kw[1])
            elif kw[0] == 'n_conditions':
                print(f'Number of conditions: {kw[1]}')
                n_conditions = int(kw[1])
            elif kw[0] == 'sample_interval':
                print(f'Sample interval: {kw[1]}')
                sample_interval = int(kw[1])
            elif kw[0] == 'path_dataset':
                print(f'Path to dataset: {kw[1]}')
                path_dataset = kw[1]

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')
    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    if trained_gan is None:
        trained_gan = True  # Use an existing GAN/Checkpoints of previous training
    if train_gan is None:
        train_gan = True  # Train the GAN in the optimization process

    # Data configuration
    if windows_slices is None:
        windows_slices = True  # Use window_slices of data with stride 1 as training samples
    diff_data = False  # Differentiate data
    std_data = False  # Standardize data
    norm_data = True  # Normalize data

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration
    opt = {
        'n_epochs': 100 if not n_epochs else n_epochs,  # number of training epochs of batch training
        'sequence_length': None if not sequence_length else sequence_length,
        # length of the sequences of the time-series data
        'seq_len_generated': 6 if not seq_len_generated else seq_len_generated,
        # length of the time-series data to-be-generated
        'hidden_dim': 128,  # Dimension of hidden layers in discriminator and generator
        'batch_size': 32 if not batch_size else batch_size,  # batch size for batch training
        'learning_rate': 1e-4 if not learning_rate else learning_rate,
        # learning rate of the generator and discriminator
        'latent_dim': 16,  # Dimension of the latent space
        'sample_interval': 10 if not sample_interval else sample_interval,  # interval between recorded samples
        'critic_iterations': 5,  # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_conditions': 1 if not n_conditions else n_conditions,  # number of conditions for conditional GAN
        'n_lstm': 2,  # number of lstm layers for lstm GAN
        'patch_size': 15 if not patch_size else patch_size,  # Patch size for the transformer GAN (tts-gan)
        'trained_gan': False if trained_gan is None else trained_gan,  # Use an existing GAN/Checkpoints of previous training
        'world_size': world_size,  # number of processes for distributed training
    }

    # Load dataset as tensor
    path = r'data/ganAverageERP.csv' if not path_dataset else path_dataset
    seq_len = opt['sequence_length'] if 'sequence_length' in opt else None
    # seq_len_2 = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
    # seq_len = seq_len_1 - seq_len_2
    dataloader = Dataloader(path, diff_data=diff_data, std_data=std_data, norm_data=norm_data)
    dataset = dataloader.get_data(sequence_length=seq_len, windows_slices=windows_slices, stride=5)
    opt['sequence_length'] = dataset.shape[1] - opt['n_conditions']

    # keep randomly 10% of the data
    # dataset = dataset[np.random.randint(0, dataset.shape[0], int(dataset.shape[0]*0.3))]

    if (opt['sequence_length']) % opt['patch_size'] != 0:
        raise ValueError(
            f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({opt['patch_size']}).")

    # Initialize generator, discriminator and trainer
    state_dict = None
    optG = None
    optD = None
    generator = Generator(seq_length=opt['seq_len_generated'],
                          latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt[
                              'seq_len_generated'],
                          patch_size=opt['patch_size'])
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size'])
    trainer = Trainer(generator, discriminator, opt)
    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # start training
        mp.spawn(run, args=(world_size, find_free_port(), trainer, dataset, trained_gan), nprocs=world_size, join=True)
    else:
        print("GAN not trained.")
