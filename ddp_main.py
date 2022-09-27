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

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    trained_gan = False             # Use an existing GAN/Checkpoints of previous training
    train_gan = True                # Train the GAN in the optimization process
    trained_embedding = False       # Use an existing embedding
    use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    diff_data = False               # Differentiate data
    std_data = False                # Standardize data
    norm_data = True                # Normalize data
    windows_slices = True           # Use window_slices of data with stride 1 as training samples

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Dataset configuration:\n\tdifferentiation: {}\n\tstandardization: {}\n\tnormalization: {}"
          .format(diff_data, std_data, norm_data))
    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    # GAN configuration
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration
    opt = {
        'n_epochs': 2,  # number of training epochs of batch training
        'sequence_length': 30,  # length of the sequences of the time-series data
        'seq_len_generated': 6,  # length of the time-series data to-be-generated
        'hidden_dim': 128,  # Dimension of hidden layers in discriminator and generator
        'batch_size': 32,  # batch size for batch training
        'learning_rate': 1e-4,  # learning rate of the generator and discriminator
        'latent_dim': 16,  # Dimension of the latent space
        'sample_interval': 10,  # interval between recorded samples
        'critic_iterations': 5,  # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_conditions': 1,  # number of conditions for conditional GAN
        'n_lstm': 2,  # number of lstm layers for lstm GAN
        'patch_size': 15  # Patch size for the transformer GAN (tts-gan)
    }

    # Load dataset as tensor
    path = r'data/ganAverageERP.csv'
    seq_len = opt['sequence_length'] if 'sequence_length' in opt else None
    # seq_len_2 = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
    # seq_len = seq_len_1 - seq_len_2
    dataloader = Dataloader(path, diff_data=diff_data, std_data=std_data, norm_data=norm_data)
    dataset = dataloader.get_data(sequence_length=seq_len, windows_slices=windows_slices, stride=5)

    if (opt['sequence_length']) % opt['patch_size'] != 0:
        raise ValueError("Sequence length must be a multiple of patch size.")

    # Initialize generator, discriminator and trainer
    state_dict = None
    optG = None
    optD = None
    generator = Generator(seq_length=opt['seq_len_generated'],
                          latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt[
                              'seq_len_generated'],
                          patch_size=opt['patch_size'])
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size'])
    trainer = Trainer(generator, discriminator, dataset, opt)
    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # start training
        mp.spawn(run, args=(world_size, find_free_port(), trainer), nprocs=world_size, join=True)
    else:
        print("GAN not trained.")
