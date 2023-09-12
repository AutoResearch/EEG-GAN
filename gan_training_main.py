import os
import sys
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader

from helpers.trainer import GANTrainer
from helpers.get_master import find_free_port
from helpers.ddp_training import run, GANDDPTrainer
from nn_architecture.models import AutoencoderGenerator, AutoencoderDiscriminator, TransformerGenerator, TransformerDiscriminator
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader
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
    """Main function of the training process."""
    default_args = system_inputs.parse_arguments(sys.argv, file='gan_training_main.py')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp']
    ddp_backend = default_args['ddp_backend']
    load_checkpoint = default_args['load_checkpoint']
    path_checkpoint = default_args['path_checkpoint']

    # Data configuration
    diff_data = False  # Differentiate data
    std_data = False  # Standardize data
    norm_data = True  # Normalize data

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    if load_checkpoint:
        print(f'Resuming training from checkpoint {path_checkpoint}.')

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration

    # Load dataset as tensor
    opt = {
        'n_epochs': default_args['n_epochs'],
        'input_sequence_length': default_args['input_sequence_length'],
        # 'seq_len_generated': default_args['seq_len_generated'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'path_autoencoder': default_args['path_autoencoder'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['conditions']) if default_args['conditions'][0] != '' else 0,
        # 'patch_size': default_args['patch_size'],
        'kw_timestep': default_args['kw_timestep'],
        'conditions': default_args['conditions'],
        'sequence_length': -1,
        'hidden_dim': default_args['hidden_dim'],  # Dimension of hidden layers in discriminator and generator
        'num_layers': default_args['num_layers'],
        'activation': default_args['activation'],
        'latent_dim': 16,  # Dimension of the latent space
        'critic_iterations': 5,  # number of iterations of the critic per generator iteration for Wasserstein GAN
        'lambda_gp': 10,  # Gradient penalty lambda for Wasserstein GAN-GP
        'world_size': world_size,  # number of processes for distributed training
        # 'multichannel': default_args['multichannel'],
        'channel_label': default_args['channel_label'],
        'norm_data': norm_data,
        'std_data': std_data,
        'diff_data': diff_data,
    }

    dataloader = Dataloader(default_args['path_dataset'],
                            kw_timestep=default_args['kw_timestep'],
                            col_label=default_args['conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data,
                            channel_label=default_args['channel_label'])
    dataset = dataloader.get_data()

    opt['channel_names'] = dataloader.channels
    opt['n_channels'] = dataset.shape[-1]
    opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    if opt['input_sequence_length'] == -1:
        opt['input_sequence_length'] = opt['sequence_length']
    opt['n_samples'] = dataset.shape[0]

    # if opt['sequence_length'] % opt['patch_size'] != 0:
    #     warnings.warn(
    #         f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
    #         f"The sequence length is padded with zeros to fit the condition.")
    #     padding = 0
    #     while (opt['sequence_length'] + padding) % default_args['patch_size'] != 0:
    #         padding += 1
    #     padding = torch.zeros((dataset.shape[0], padding))
    #     dataset = torch.cat((dataset, padding), dim=1)
    #     opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]

    latent_dim_in = opt['latent_dim'] + opt['n_conditions'] + opt['n_channels'] if opt['input_sequence_length'] > 0 else \
    opt['latent_dim'] + opt['n_conditions']
    channel_in_disc = opt['n_channels'] + opt['n_conditions']
    sequence_length_generated = opt['sequence_length'] - opt['input_sequence_length'] if opt['input_sequence_length'] != opt['sequence_length'] else opt['sequence_length']

    # Initialize generator, discriminator and trainer
    if opt['path_autoencoder'] == '':
        # no autoencoder defined -> use transformer GAN
        generator = TransformerGenerator(latent_dim=latent_dim_in,
                                         channels=opt['n_channels'],
                                         seq_len=sequence_length_generated)
        discriminator = TransformerDiscriminator(channels=channel_in_disc,
                                                 hidden_dim=opt['hidden_dim'])
    else:
        # initialize the autoencoder
        seq_length=dataset.shape[1]-len(opt['conditions'])
        ae_dict = torch.load(default_args['path_autoencoder'], map_location=torch.device('cpu'))
        if ae_dict['configuration']['target'] == 'channels':
            autoencoder = TransformerAutoencoder(input_dim=ae_dict['configuration']['input_dim'],
                                        output_dim=ae_dict['configuration']['output_dim'],
                                        target=TransformerAutoencoder.TARGET_CHANNELS,
                                        hidden_dim=ae_dict['configuration']['hidden_dim'],
                                        num_layers=ae_dict['configuration']['num_layers'],
                                        num_heads=ae_dict['configuration']['num_heads'],).to(device)
        elif ae_dict['configuration']['target'] == 'time':
            autoencoder = TransformerAutoencoder(input_dim=seq_length,
                                        output_dim=ae_dict['configuration']['timeseries_out'],
                                        target=TransformerAutoencoder.TARGET_TIMESERIES,
                                        hidden_dim=ae_dict['configuration']['hidden_dim'],
                                        num_layers=ae_dict['configuration']['num_layers'],
                                        num_heads=ae_dict['configuration']['num_heads'], ).to(device)
        elif ae_dict['configuration']['target'] == 'full':
            autoencoder = TransformerDoubleAutoencoder(input_dim=ae_dict['configuration']['input_dim'],
                                                output_dim=ae_dict['configuration']['output_dim'],
                                                sequence_length=seq_length,
                                                output_dim_2=ae_dict['configuration']['output_dim_2'],
                                                hidden_dim=ae_dict['configuration']['hidden_dim'],
                                                num_layers=ae_dict['configuration']['num_layers'],
                                                num_heads=ae_dict['configuration']['num_heads'],).to(device)
        else:
            raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
        consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
        autoencoder.load_state_dict(ae_dict['model'])
        # freeze the autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()

        # if prediction or seq2seq, adjust latent_dim_in to encoded input size
        if opt['input_sequence_length'] != 0:
            new_input_dim = autoencoder.output_dim if not hasattr(autoencoder, 'output_dim_2') else autoencoder.output_dim*autoencoder.output_dim_2
            latent_dim_in += new_input_dim - autoencoder.input_dim
        generator = AutoencoderGenerator(latent_dim=latent_dim_in,
                                         autoencoder=autoencoder,
                                         num_layers=opt['num_layers'],
                                         hidden_dim=opt['hidden_dim'],
                                         activation=opt['activation'],)
        discriminator = AutoencoderDiscriminator(input_dim=channel_in_disc,
                                                 autoencoder=autoencoder,
                                                 num_layers=opt['num_layers'],
                                                 hidden_dim=opt['hidden_dim'],
                                                 activation=opt['activation'],)

        if isinstance(generator, AutoencoderGenerator) and opt['input_sequence_length'] == 0:
            # if input_sequence_length is 0, do not decode the generator output during training
            generator.decode_output(False)

        if isinstance(discriminator, AutoencoderDiscriminator) and opt['input_sequence_length'] == 0:
            # if input_sequence_length is 0, do not encode the discriminator input during training
            discriminator.encode_input(False)

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
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        mp.spawn(run,
                 args=(world_size, find_free_port(), ddp_backend, trainer, opt),
                 nprocs=world_size, join=True)
    else:
        trainer = GANTrainer(generator, discriminator, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        dataset = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=True)
        gen_samples = trainer.training(dataset)

        # save final models, optimizer states, generated samples, losses and configuration as final result
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gan_{trainer.epochs}ep_' + timestamp + '.pt'
        trainer.save_checkpoint(path_checkpoint=os.path.join(path, filename), samples=gen_samples)

        generator = trainer.generator
        discriminator = trainer.discriminator

        print("GAN training finished.")
        print(f"Model states and generated samples saved to file {os.path.join(path, filename)}.")

        return generator, discriminator, opt, gen_samples


if __name__ == '__main__':
    main()
