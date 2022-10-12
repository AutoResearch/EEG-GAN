import os
import sys

import numpy as np
import pandas as pd
import torch

import system_inputs
from trainer import Trainer
from models import TtsGenerator, TtsGeneratorFiltered
from dataloader import Dataloader


if __name__ == '__main__':
    # get default system arguments
    system_args = system_inputs.default_inputs_generate_samples()
    default_args = {}
    for key, value in system_args.items():
        # value = [type, description, default value]
        default_args[key] = value[2]

    file, path_samples, sequence_length_total, filter_generator, condition, num_samples_total, num_samples_parallel, kw_timestep_dataset\
        = None, None, None, None, None, None, None, None

    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')
    for arg in sys.argv:
        if '.py' not in arg:
            if arg == 'help':
                helper = system_inputs.HelperGenerateSamples('generate_samples_main.py', system_inputs.default_inputs_generate_samples())
                helper.print_table()
                helper.print_help()
                exit()
            elif arg == 'filter_generator':
                print('Using bandpass-filtered generator')
                filter_generator = True
            elif '=' in arg:
                kw = arg.split('=')
                if kw[0] == 'file':
                    print(f'Using checkpoint file: {kw[1]}')
                    file = kw[1]
                elif kw[0] == 'condition':
                    print(f'Given condition is: {kw[1]}')
                    condition = system_inputs.return_list(kw[1])
                elif kw[0] == 'num_samples_total':
                    print(f'Total number of samples: {kw[1]}')
                    num_samples_total = int(kw[1])
                elif kw[0] == 'num_samples_parallel':
                    print(f'Number of samples computed in parallel: {kw[1]}')
                    num_samples_parallel = int(kw[1])
                elif kw[0] == 'path_samples':
                    print(f'Path to save samples: {kw[1]}')
                    path_samples = kw[1]
                elif kw[0] == 'sequence_length_total':
                    print(f'Total sequence length: {kw[1]}')
                    sequence_length_total = int(kw[1])
                elif kw[0] == 'kw_timestep_dataset':
                    print(f'Keyword for timestep in dataset: {kw[1]}')
                    kw_timestep_dataset = kw[1]
                else:
                    print(f'Keyword {kw[0]} not recognized. Please use the keyword "help" to see the available arguments.')
            else:
                print(f'Keyword {arg} not recognized. Please use the keyword "help" to see the available arguments.')

    filter_generator = default_args['filter_generator'] if filter_generator is None else filter_generator
    sequence_length_total = default_args['sequence_length_total'] if sequence_length_total is None else sequence_length_total
    num_samples_total = default_args['num_samples_total'] if num_samples_total is None else num_samples_total
    num_samples_parallel = default_args['num_samples_parallel'] if num_samples_parallel is None else num_samples_parallel
    kw_timestep_dataset = default_args['kw_timestep_dataset'] if kw_timestep_dataset is None else kw_timestep_dataset
    condition = default_args['condition'] if condition is None else condition
    if not isinstance(condition, list):
        condition = [condition]
    file = default_args['file'] if file is None else file
    if file.split(os.path.sep)[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    path_samples = default_args['path_samples'] if path_samples is None else path_samples
    if path_samples.split(os.path.sep)[0] == path_samples:
        # use default path if no path is given
        path = 'generated_samples'
        path_samples = os.path.join(path, path_samples)

    state_dict = torch.load(file, map_location='cpu')

    # load model/training configuration
    filename_dataset = state_dict['configuration']['path_dataset']
    n_conditions = state_dict['configuration']['n_conditions']
    latent_dim = state_dict['configuration']['latent_dim']
    sequence_length = state_dict['configuration']['sequence_length']
    seq_len_gen = state_dict['configuration']['sequence_length_generated']
    patch_size = state_dict['configuration']['patch_size']
    seq_len_cond = sequence_length - seq_len_gen

    # get the sequence length from the dataset
    if sequence_length_total == -1:
        cols = pd.read_csv(filename_dataset, header=0, nrows=0).columns.tolist()
        for x in cols:
            if kw_timestep_dataset in x:
                sequence_length_total += 1

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator
    print("Initializing generator...")
    if not filter_generator:
        generator = TtsGenerator(seq_length=seq_len_gen,
                                 latent_dim=latent_dim + n_conditions + seq_len_cond,
                                 patch_size=patch_size).to(device)
    else:
        generator = TtsGeneratorFiltered(seq_length=seq_len_gen,
                                         latent_dim=latent_dim + n_conditions + seq_len_cond,
                                         patch_size=patch_size).to(device)
    generator.eval()

    # load generator weights
    generator.load_state_dict(state_dict['generator'])

    # create condition labels
    cond_labels = torch.zeros((num_samples_parallel, n_conditions))
    for n in range(num_samples_parallel):
        for i, x in enumerate(condition):
            if x == -1:
                # random condition (works currently only for binary conditions)
                x = np.random.randint(0, 2)
            cond_labels[n, i] = x

    # init sequence for windows_slices
    sequence = torch.zeros((num_samples_parallel, seq_len_cond))

    # generate samples
    num_sequences = int(np.floor(num_samples_total / num_samples_parallel))
    all_samples = np.zeros((num_samples_parallel * num_sequences, sequence_length_total + n_conditions))
    print("Generating samples...")

    for i in range(num_sequences):
        print(f"Generating sequence {i+1} of {num_sequences}...")
        while sequence.shape[1] < sequence_length_total + seq_len_cond:
            # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
            z = Trainer.sample_latent_variable(batch_size=num_samples_parallel, latent_dim=latent_dim)
            z = torch.cat((z, cond_labels, sequence[:, -seq_len_cond:]), dim=1).type(torch.FloatTensor).to(device)
            samples = generator(z)
            sequence = torch.cat((sequence, samples.view(num_samples_parallel, -1)), dim=1)
        sequence = sequence[:, seq_len_cond:seq_len_cond+sequence_length_total]
        sequence = torch.cat((cond_labels, sequence), dim=1)
        all_samples[i * num_samples_parallel:(i + 1) * num_samples_parallel, :] = sequence.detach().cpu().numpy()

    # save samples
    print("Saving samples...")
    path = 'generated_samples'
    pd.DataFrame(all_samples).to_csv(path_samples)

    print("Generated samples were saved to " + path_samples)
