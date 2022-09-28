import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np

from trainer import Trainer
from models import TtsDiscriminator, TtsGenerator as Generator  #, TtsGeneratorFiltered
from dataloader import Dataloader
from EmbeddingNet import Encoder, Decoder, Trainer as EmbeddingNetTrainer

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
    """Main function of the training process. Arguments can be given in the command line.
    :arg path_dataset: path to the dataset (Standard: 'data/ganAverageERP.csv')
    :arg n_epochs: number of epochs to train the GAN
    :arg sequence_length: total length of the time-series data
    :arg seq_len_generated: length of the time-series data to generate
    :arg trained_gan: use a pre-trained GAN
    :arg train_gan: train the GAN
    :arg windows_slices: use window slices
    :arg patch_size: patch size of the transformer.
    :arg batch_size: batch size
    :arg learning_rate: learning rate
    :arg n_conditions: number of conditions
    :arg sample_interval: interval between samples
    """


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
        trained_gan = False              # Use an existing GAN/Checkpoints of previous training
    if train_gan is None:
        train_gan = True                # Train the GAN in the optimization process
    trained_embedding = False       # Use an existing embedding
    use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    diff_data = False               # Differentiate data
    std_data = False                # Standardize data
    norm_data = True                # Normalize data
    if windows_slices is None:
        windows_slices = False           # Use window_slices of data with stride 1 as training samples

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GAN configuration
    opt = {
            'n_epochs': 100 if not n_epochs else n_epochs,  # number of training epochs of batch training
            'sequence_length': 90 if not sequence_length else sequence_length,  # length of the sequences of the time-series data
            'seq_len_generated': 10 if not seq_len_generated else seq_len_generated,  # length of the time-series data to-be-generated
            'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
            'batch_size': 32 if not batch_size else batch_size,  # batch size for batch training
            'learning_rate': 1e-4 if not learning_rate else learning_rate,  # learning rate of the generator and discriminator
            'latent_dim': 16,           # Dimension of the latent space
            'sample_interval': 100 if not sample_interval else sample_interval,  # interval between recorded samples
            'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
            'n_conditions': 1 if not n_conditions else n_conditions,  # number of conditions for conditional GAN
            'n_lstm': 2,                # number of lstm layers for lstm GAN
            'patch_size': 15 if not patch_size else patch_size  # Patch size for the transformer GAN (tts-gan)
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

    if opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({opt['patch_size']}).\n"
                      f"The sequence length is padded with zeros to fit the condition.")
        while opt['sequence_length'] % opt['patch_size'] != 0:
            dataset = torch.cat((dataset, torch.zeros(dataset.shape[0], 1)), dim=-1)
            opt['sequence_length'] += 1

    # Embedding network to reduce the dimension of time-series data
    # not tested yet
    if use_embedding:
        # Use pretrained embedding
        if trained_embedding:
            # load encoder
            encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
            encoder_weights = torch.load(r'trained_models\embedding_encoder.pt')
            encoder.load_state_dict(encoder_weights)
            # load decoder
            decoder = Decoder(output_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
            decoder_weights = torch.load(r'trained_models\\embedding_decoder.pt')
            decoder.load_state_dict(decoder_weights)
            print('Loaded pretrained embedding.')
        else:
            # train embedding
            print('Training embedding...')
            encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
            decoder = Decoder(signals=1, conditions=1, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'],
                              seq_len=dataset.shape[1]-opt['n_conditions'])
            embedding_trainer = EmbeddingNetTrainer(encoder, decoder, opt)
            encoder, decoder, emb_samples, losses = embedding_trainer.train(dataset)
            print('Finished training embedding.')
            plt.plot(losses)
            plt.show()

            # save embedding
            # pickle emb_samples
            # with open('emb_samples.pkl', 'wb') as f:
            #     pickle.dump(emb_samples, f)
            df = pd.DataFrame(emb_samples, columns=None, index=None).T
            torch.save(encoder.state_dict(), 'trained_models/encoder_' + timestamp + '.pt')
            torch.save(decoder.state_dict(), 'trained_models/decoder_' + timestamp + '.pt')

    # Initialize generator, discriminator and trainer
    state_dict = None
    optG = None
    optD = None
    generator = Generator(seq_length=opt['seq_len_generated'],
                          latent_dim=opt['latent_dim']+opt['n_conditions']+opt['sequence_length']-opt['seq_len_generated'],
                          patch_size=opt['patch_size']).to(device)
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size']).to(device)
    trainer = Trainer(generator, discriminator, opt, optimizer_generator=optG, optimizer_discriminator=optD)
    print("Generator and discriminator initialized.")

    if trained_gan:
        # Use pretrained generator and discriminator

        # check if checkpoint-file exists
        if os.path.isfile(r'trained_models\checkpoint.pt'):
            # load state_dicts
            state_dict = torch.load(r'trained_models\checkpoint_01.pt', map_location=trainer.device)
            trainer.generator.load_state_dict(state_dict['generator'])
            trainer.discriminator.load_state_dict(state_dict['discriminator'])
            trainer.generator_optimizer.load_state_dict(state_dict['optimizer_generator'])
            trainer.discriminator_optimizer.load_state_dict(state_dict['optimizer_discriminator'])
            print("Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. If you do not wish to continue training, set trained_gan to False.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # GAN-Training
        print('\n-----------------------------------------')
        print("Training GAN...")
        print('-----------------------------------------\n')
        generator, discriminator, gen_samples = trainer.training(dataset)

        print("GAN training finished.")
        print("Generated samples saved to file.")
        print("Model states saved to file.")
    else:
        print("GAN not trained.")
    