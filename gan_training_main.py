import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np

from trainer import Trainer
from models import TtsDiscriminator, TtsGenerator, TtsGeneratorFiltered
import torch.multiprocessing as mp

from get_master import find_free_port
from ddp_training import run, DDPTrainer

from models import TtsDiscriminator, TtsGenerator, TtsGeneratorFiltered
from dataloader import Dataloader
import system_inputs

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
    :arg load_checkpoint: load a pre-trained GAN from a checkpoint
    :arg path_checkpoint: path to the pre-trained GAN
    :arg train_gan: train the GAN
    :arg windows_slices: use window slices
    :arg patch_size: patch size of the transformer.
    :arg batch_size: batch size
    :arg learning_rate: learning rate
    :arg n_conditions: number of conditions
    :arg sample_interval: interval between samples
    """

    # get default system arguments
    system_args = system_inputs.default_inputs_main()
    default_args = {}
    for key, value in system_args.items():
        # value = [type, description, default value]
        default_args[key] = value[2]

    # Get system arguments
    ddp, n_epochs, sequence_length, seq_len_generated, load_checkpoint, path_checkpoint, train_gan, \
        windows_slices, patch_size, batch_size, learning_rate, sample_interval, n_conditions, path_dataset, \
        filter_generator, ddp_backend = \
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')
    for arg in sys.argv:
        if arg == 'help':
            helper = system_inputs.HelperMain('gan_training_main.py', system_args)
            helper.print_table()
            helper.print_help()
            exit()
        elif arg == 'ddp':
            ddp = True
        elif arg == 'load_checkpoint':
            print('Loading checkpoint')
            load_checkpoint = True
        elif arg == 'train_gan':
            print('Training GAN')
            train_gan = True
        elif arg == 'windows_slices':
            print('Using window slices')
            windows_slices = True
        elif arg == 'filter_generator':
            print('Using low-pass-filtered generator')
            filter_generator = True
        elif '=' in arg:
            kw = arg.split('=')
            if kw[0] == 'ddp':
                print(f'Use distributed data parallel training: {kw[1]}')
                ddp = kw[1] == 'True'
            if kw[0] == 'n_epochs':
                print(f'Number of epochs: {kw[1]}')
                n_epochs = int(kw[1])
            elif kw[0] == 'sequence_length':
                print(f'Total sequence length: {kw[1]}')
                sequence_length = int(kw[1])
            elif kw[0] == 'seq_len_generated':
                print(f'Sequence length to generate: {kw[1]}')
                seq_len_generated = int(kw[1])
            elif kw[0] == 'path_checkpoint':
                print(f'Path to checkpoint: {kw[1]}')
                path_checkpoint = kw[1]
            elif kw[0] == 'load_checkpoint':
                print(f'Use checkpoint: {kw[1]}')
                load_checkpoint = kw[1] == 'True'
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
            elif kw[0] == 'ddp_backend':
                print(f'Distributed data parallel backend: {kw[1]}')
                ddp_backend = kw[1]
            elif kw[0] == 'filter_generator':
                print(f'Filter generator: {kw[1]}')
                filter_generator = kw[1] == 'True'
            else:
                print(f'Argument {kw[1]} not recognized. Use the keyword "help" to see the available arguments.')
        else:
            print(f'Keyword {arg} not recognized. Please use the keyword "help" to see the available arguments.')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp'] if ddp is None else ddp
    ddp_backend = default_args['ddp_backend'] if ddp_backend is None else ddp_backend
    load_checkpoint = default_args['load_checkpoint'] if load_checkpoint is None else load_checkpoint
    path_checkpoint = default_args['path_checkpoint'] if path_checkpoint is None else path_checkpoint
    train_gan = default_args['train_gan'] if train_gan is None else train_gan
    # trained_embedding = False       # Use an existing embedding
    # use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    if windows_slices is None:
        # Use window_slices of data with stride 1 as training samples
        windows_slices = default_args['windows_slices']
    diff_data = False               # Differentiate data
    std_data = False                # Standardize data
    norm_data = True                # Normalize data
    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration
    opt = {
        'n_epochs': default_args['n_epochs'] if n_epochs is None else n_epochs,
        'sequence_length': default_args['sequence_length'] if sequence_length is None else sequence_length,
        'seq_len_generated': default_args['seq_len_generated'] if seq_len_generated is None else seq_len_generated,
        'load_checkpoint': default_args['load_checkpoint'] if load_checkpoint is None else load_checkpoint,
        'path_checkpoint': default_args['path_checkpoint'] if path_checkpoint is None else path_checkpoint,
        'path_dataset': default_args['path_dataset'] if path_dataset is None else path_dataset,
        'batch_size': default_args['batch_size'] if batch_size is None else batch_size,
        'learning_rate': default_args['learning_rate'] if learning_rate is None else learning_rate,
        'sample_interval': default_args['sample_interval'] if sample_interval is None else sample_interval,
        'n_conditions': default_args['n_conditions'] if n_conditions is None else n_conditions,
        'patch_size': default_args['patch_size'] if patch_size is None else patch_size,
        'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
        'latent_dim': 16,           # Dimension of the latent space
        'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_lstm': 2,                # number of lstm layers for lstm GAN
        'world_size': world_size,   # number of processes for distributed training
    }

    # Load dataset as tensor
    path = opt['path_dataset'] if 'path_dataset' in opt else default_args['path_dataset']
    seq_len = opt['sequence_length'] if 'sequence_length' in opt else None
    # seq_len_2 = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
    # seq_len = seq_len_1 - seq_len_2
    dataloader = Dataloader(path, diff_data=diff_data, std_data=std_data, norm_data=norm_data)
    dataset = dataloader.get_data(sequence_length=seq_len, windows_slices=windows_slices, stride=5, pre_pad=opt['sequence_length']-opt['seq_len_generated'])
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
    # if use_embedding:
    #     # Use pretrained embedding
    #     if trained_embedding:
    #         # load encoder
    #         encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         encoder_weights = torch.load(r'trained_models\embedding_encoder.pt')
    #         encoder.load_state_dict(encoder_weights)
    #         # load decoder
    #         decoder = Decoder(output_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         decoder_weights = torch.load(r'trained_models\\embedding_decoder.pt')
    #         decoder.load_state_dict(decoder_weights)
    #         print('Loaded pretrained embedding.')
    #     else:
    #         # train embedding
    #         print('Training embedding...')
    #         encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         decoder = Decoder(signals=1, conditions=1, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'],
    #                           seq_len=dataset.shape[1]-opt['n_conditions'])
    #         embedding_trainer = EmbeddingNetTrainer(encoder, decoder, opt)
    #         encoder, decoder, emb_samples, losses = embedding_trainer.train(dataset)
    #         print('Finished training embedding.')
    #         plt.plot(losses)
    #         plt.show()
    #
    #         # save embedding
    #         # pickle emb_samples
    #         # with open('emb_samples.pkl', 'wb') as f:
    #         #     pickle.dump(emb_samples, f)
    #         df = pd.DataFrame(emb_samples, columns=None, index=None).T
    #         torch.save(encoder.state_dict(), 'trained_models/encoder_' + timestamp + '.pt')
    #         torch.save(decoder.state_dict(), 'trained_models/decoder_' + timestamp + '.pt')

    # Initialize generator, discriminator and trainer

    if filter_generator is None:
        generator = TtsGenerator(seq_length=opt['seq_len_generated'],
                                 latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt['seq_len_generated'],
                                 patch_size=opt['patch_size'])
    else:
        generator = TtsGeneratorFiltered(seq_length=opt['seq_len_generated'],
                                         latent_dim=opt['latent_dim']+opt['n_conditions']+opt['sequence_length']-opt['seq_len_generated'],
                                         patch_size=opt['patch_size'])
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size'])
    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # GAN-Training
        print('\n-----------------------------------------')
        print("Training GAN...")
        print('-----------------------------------------\n')
        if ddp:
            trainer = DDPTrainer(generator, discriminator, opt)
            mp.spawn(run, args=(world_size, find_free_port(), ddp_backend, trainer, dataset),
                     nprocs=world_size, join=True)
        else:
            trainer = Trainer(generator, discriminator, opt)
            gen_samples = trainer.training(dataset)

            # save final models, optimizer states, generated samples, losses and configuration as final result
            path = 'trained_models'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'state_dict_{trainer.epochs}ep_' + timestamp + '.pt'
            trainer.save_checkpoint(path_checkpoint=os.path.join(path, filename), generated_samples=gen_samples)

        print("GAN training finished.")
        print("Generated samples saved to file.")
        print("Model states saved to file.")
    else:
        print("GAN not trained.")
    