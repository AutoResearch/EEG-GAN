from datetime import datetime
import torch

import torch.multiprocessing as mp

from get_master import find_free_port
from ddp_training import run, DDPTrainer as Trainer
from models import TtsDiscriminator, TtsGenerator#, TtsGeneratorFiltered
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

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    trained_gan = True             # Use an existing GAN/Checkpoints of previous training
    train_gan = True                # Train the GAN in the optimization process
    trained_embedding = False       # Use an existing embedding
    use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    diff_data = False               # Differentiate data
    std_data = False                # Standardize data
    norm_data = True                # Normalize data

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Dataset configuration:\n\tdifferentiation: {}\n\tstandardization: {}\n\tnormalization: {}"
          .format(diff_data, std_data, norm_data))
    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    # GAN configuration
    world_size = 2  #torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
    opt = {
            'n_epochs': 2,              # number of training epochs of batch training
            'sample_interval': 100,  # interval between recorded samples
            'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
            'batch_size': 8,            # batch size for batch training
            'learning_rate': 1e-4,      # learning rate of the generator and discriminator
            'latent_dim': 16,           # Dimension of the latent space
            'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
            'n_conditions': 1,          # number of conditions for conditional GAN
            'n_lstm': 2,                # number of lstm layers for lstm GAN
            'patch_size': 15,           # Patch size for the transformer GAN (tts-gan)
            'trained_gan': trained_gan, # Use an existing GAN/Checkpoints of previous training
        }

    # Load dataset as tensor
    path = r'data/ganAverageERP_mini.csv'
    dataloader = Dataloader(path, diff_data, std_data, norm_data)
    dataset = dataloader.get_data()
    # make sequence length of dataset dividable by opt['patch_size'] by padding with zeros
    while (dataset.shape[1] - opt['n_conditions']) % opt['patch_size'] != 0:
        padding = torch.zeros(dataset.shape[0], 1)
        dataset = torch.cat((dataset, padding), dim=1)

    # Initialize generator, discriminator and trainer
    generator = TtsGenerator(seq_length=dataset.shape[1]-opt['n_conditions'], latent_dim=opt['latent_dim']+opt['n_conditions'])
    discriminator = TtsDiscriminator(seq_length=dataset.shape[1]-opt['n_conditions'])
    trainer = Trainer(generator, discriminator, dataset, opt)
    print("Generator, discriminator and trainer initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # start training
        mp.spawn(run, args=(world_size, find_free_port(), trainer), nprocs=world_size, join=True)
    else:
        print("GAN not trained.")
