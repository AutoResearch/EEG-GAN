import os
import sys

import numpy as np
import pandas as pd
import torch

from helpers import system_inputs
from helpers.trainer import Trainer
from nn_architecture.models import TtsGenerator, TtsGeneratorFiltered

if __name__ == '__main__':

    # sys.argv = ["file=gan_multiCond.pt", "average=10", "conditions=-1,-2,0.5"]
    default_args = system_inputs.parse_arguments(sys.argv, file='generate_samples_main.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    sequence_length_total = default_args['sequence_length_total']
    num_samples_total = default_args['num_samples_total']
    num_samples_parallel = default_args['num_samples_parallel']
    kw_timestep_dataset = default_args['kw_timestep_dataset']
    average_over = default_args['average']
    all_cond_per_z = default_args['all_cond_per_z']

    condition = default_args['conditions']
    if not isinstance(condition, list):
        condition = [condition]
    file = default_args['file']
    if file.split(os.path.sep)[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    path_samples = default_args['path_samples']
    if path_samples == 'None':
        # Use checkpoint filename as path
        path_samples = os.path.basename(file).split('.')[0] + '.csv'
    if path_samples.split(os.path.sep)[0] == path_samples:
        # use default path if no path is given
        path = 'generated_samples'
        path_samples = os.path.join(path, path_samples)

    state_dict = torch.load(file, map_location='cpu')

    # load model/training configuration
    filename_dataset = state_dict['configuration']['path_dataset']
    n_conditions = state_dict['configuration']['n_conditions']
    n_channels = state_dict['configuration']['n_channels']
    channel_names = state_dict['configuration']['channel_names']
    latent_dim = state_dict['configuration']['latent_dim']
    sequence_length = state_dict['configuration']['sequence_length']
    seq_len_gen = state_dict['configuration']['sequence_length_generated']
    patch_size = state_dict['configuration']['patch_size']
    filter_generator = True if state_dict['configuration']['generator'] == 'TtsGeneratorFiltered' else False
    seq_len_cond = sequence_length - seq_len_gen

    # get the sequence length from the dataset
    if sequence_length_total == -1:
        cols = pd.read_csv(filename_dataset, header=0, nrows=0).columns.tolist()
        sequence_length_total = len([index for index in range(len(cols)) if kw_timestep_dataset in cols[index]])

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator
    print("Initializing generator...")
    if not filter_generator:
        generator = TtsGenerator(seq_length=seq_len_gen,
                                 latent_dim=latent_dim + n_conditions + seq_len_cond,
                                 patch_size=patch_size,
                                 channels=n_channels).to(device)
    else:
        generator = TtsGeneratorFiltered(seq_length=seq_len_gen,
                                         latent_dim=latent_dim + n_conditions + seq_len_cond,
                                         patch_size=patch_size,
                                         channels=n_channels).to(device)
    generator.eval()

    # load generator weights
    generator.load_state_dict(state_dict['generator'])

    # check given conditions that they are numeric
    for i, x in enumerate(condition):
        if x == -1 or x == -2:
            continue
        else:
            try:
                condition[i] = float(x)
            except ValueError:
                raise ValueError(f"Condition {x} is not numeric.")

    # create condition labels if conditions are given but differ from number of conditions in model
    if n_conditions != len(condition):
        if n_conditions > len(condition) and len(condition) == 1 and condition[0] == -1:
            # if only one condition is given and it is -1, then all conditions are set to -1
            condition = condition * n_conditions
        else:
            raise ValueError(f"Number of conditions in model (={n_conditions}) does not match number of conditions given ={len(condition)}.")

    cond_labels = torch.zeros((num_samples_parallel, n_conditions)).to(device)
    #JOSHUA
    '''
    for n in range(num_samples_parallel):
        for i, x in enumerate(condition):
            if x == -1:
                # random condition (works currently only for binary conditions)
                # cond_labels[n, i] = np.random.randint(0, 2)  # TODO: Channel recovery: Maybe better - random conditions for each entry
                cond_labels[n, i] = 0 if n % 2 == 0 else 1  # TODO: Currently all conditions of one row are the same (0 or 1)
     '''

    #JOSHUA'S CODE
    '''
    for n in range(num_samples_parallel):
        for i, x in enumerate(condition):
            if x == -1:
                # random condition (works currently only for binary conditions)
                # cond_labels[n, i] = np.random.randint(0, 2)  # TODO: Channel recovery: Maybe better - random conditions for each entry
                cond_labels[n, i] = 0 if n % 2 == 0 else 1  # TODO: Currently all conditions of one row are the same (0 or 1)
     '''

    def get_condition_label(condition, num_samples_parallel):
        # create labels for generator according to given conditions
        # if condition is -1, generate random binary conditions
        # if condition is -2, generate random continuous conditions between 0 and 1
        # if conditions are given, use them
        for n in range(num_samples_parallel):
            for i, x in enumerate(condition):
                if x == -1:
                    # random binary condition
                    cond_labels[n, i] = np.random.randint(0, 2)
                    # cond_labels[n, i] = 0 if n % 2 == 0 else 1  # TODO: Currently all conditions of one row are the same (0 or 1)
                elif x == -2:
                    # random continuous condition
                    cond_labels[n, i] = np.random.uniform(0, 1)
                else:
                    # given condition
                    cond_labels[n, i] = x
        return cond_labels

    # generate samples
    num_sequences = int(np.floor(num_samples_total / num_samples_parallel))
    print("Generating samples...")

    # Generation of samples begins
    if n_channels == 1:
        all_samples = np.zeros((num_samples_parallel * num_sequences * n_channels, sequence_length_total + n_conditions))
        for i in range(num_sequences):
            print(f"Generating sequence {i+1} of {num_sequences}...")
            # init sequence for windows_slices
            sequence = torch.zeros((num_samples_parallel, seq_len_cond)).to(device)
            samples = torch.zeros((num_samples_parallel, n_channels, seq_len_gen)).to(device)
            while sequence.shape[1] < sequence_length_total + seq_len_cond:
                samples = torch.zeros((num_samples_parallel, n_channels, seq_len_gen)).to(device)
                z = torch.zeros((num_samples_parallel, latent_dim)).to(device)
                if all_cond_per_z:
                    for j in range(0, num_samples_parallel-1, 2):
                        # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
                        latent_var = Trainer.sample_latent_variable(batch_size=average_over, latent_dim=latent_dim, device=device).mean(dim=0)
                        z[j] = latent_var
                        z[j+1] = latent_var
                else:
                    # For normal sample generation - use this loop
                    for j in range(num_samples_parallel):
                        # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
                        latent_var = Trainer.sample_latent_variable(batch_size=average_over, latent_dim=latent_dim, device=device).mean(dim=0)
                        z[j] = latent_var
                z = torch.cat((z, cond_labels, sequence[:, -seq_len_cond:]), dim=1).type(torch.FloatTensor).to(device)
                samples += generator(z).view(num_samples_parallel, n_channels, -1)
                sequence = torch.cat((sequence, samples[:, 0, :]), dim=1)
            sequence = sequence[:, seq_len_cond:seq_len_cond+sequence_length_total]
            sequence = torch.cat((cond_labels, sequence), dim=1)
            all_samples[i * num_samples_parallel:(i + 1) * num_samples_parallel, 0, :] = sequence.detach().cpu().numpy()
    else:
        all_samples = np.zeros((num_samples_parallel * num_sequences * n_channels, sequence_length_total + n_conditions + 1))
        for i in range(num_sequences):
            print(f"Generating sequence {i+1} of {num_sequences}...")
            # init sequence for windows_slices
            samples = torch.zeros((num_samples_parallel, n_channels, seq_len_gen+n_conditions+1)).to(device)
            z = torch.zeros((num_samples_parallel, latent_dim)).to(device)

            #JOSHUA
            '''
            # For normal sample generation - use this loop
            for j in range(num_samples_parallel):
                z[j] = Trainer.sample_latent_variable(batch_size=average_over, latent_dim=latent_dim, device=device).mean(dim=0)
            z = torch.cat((z, cond_labels), dim=1).type(torch.FloatTensor).to(device)
            
            #Create electrode labels
            chan_labels = torch.linspace(0, n_channels-1, n_channels)

            samples[:, :, n_conditions+1:] += generator(z).view(num_samples_parallel, n_channels, -1)
            samples[:, :, :n_conditions] = cond_labels.repeat(1, n_channels).view(-1, n_channels, 1)
            samples[:, :, n_conditions:n_conditions+1] = chan_labels.repeat(cond_labels.shape[0]).view(-1,n_channels,1)
            all_samples[i * num_samples_parallel*n_channels:(i + 1) * num_samples_parallel*n_channels] = samples.view(-1, samples.shape[-1]).detach().cpu().numpy()
            '''
            if all_cond_per_z:
                for j in range(0, num_samples_parallel-1, 2):
                    # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
                    latent_var = Trainer.sample_latent_variable(batch_size=average_over, latent_dim=latent_dim, device=device).mean(dim=0)
                    z[j] = latent_var
                    z[j+1] = latent_var
            else:
                # For normal sample generation - use this loop
                # for j in range(num_samples_parallel):
                    # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
                z = Trainer.sample_latent_variable(batch_size=num_samples_parallel, latent_dim=latent_dim, device=device)#.mean(dim=0)
                    # z[j] = latent_var
            cond_labels = get_condition_label(condition, num_samples_parallel)
            z = torch.cat((z, cond_labels, sequence[:, -seq_len_cond:]), dim=1).type(torch.FloatTensor).to(device)
            samples += generator(z).view(num_samples_parallel, -1)
            sequence = torch.cat((sequence, samples), dim=1)
        sequence = sequence[:, seq_len_cond:seq_len_cond+sequence_length_total]
        sequence = torch.cat((cond_labels, sequence), dim=1)
        all_samples[i * num_samples_parallel:(i + 1) * num_samples_parallel, :] = sequence.detach().cpu().numpy()

    # save samples
    print("Saving samples...")
    df = pd.DataFrame(all_samples)
    df.insert(loc=n_conditions, column='Electrode', value='')
    for i, channel_name in enumerate(channel_names):
        df.loc[df.index % n_channels == i, 'Electrode'] = channel_name
    df.to_csv(path_samples, index=False)
    pd.DataFrame(all_samples).to_csv(path_samples, index=False)

    print("Generated samples were saved to " + path_samples)
