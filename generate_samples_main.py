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

    # sys.argv = ["file=sd_len100_fullseq_9300ep.pt", "conditi"]
    default_args = system_inputs.parse_arguments(sys.argv, file='generate_samples_main.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    sequence_length_total = default_args['sequence_length_total']
    num_samples_total = default_args['num_samples_total']
    num_samples_parallel = default_args['num_samples_parallel']
    kw_timestep_dataset = default_args['kw_timestep_dataset']
    condition = default_args['conditions']
    if not isinstance(condition, list):
        condition = [condition]
    file = default_args['file']
    if file.split(os.path.sep)[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    path_samples = default_args['path_samples']
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
                                 patch_size=patch_size).to(device)
    else:
        generator = TtsGeneratorFiltered(seq_length=seq_len_gen,
                                         latent_dim=latent_dim + n_conditions + seq_len_cond,
                                         patch_size=patch_size).to(device)
    generator.eval()

    # load generator weights
    generator.load_state_dict(state_dict['generator'])

    # create condition labels
    if n_conditions != len(condition):
        raise ValueError(f"Number of conditions in model (={n_conditions}) does not match number of conditions given ={len(condition)}.")

    cond_labels = torch.zeros((num_samples_parallel, n_conditions)).to(device)

    for n in range(num_samples_parallel):
        for i, x in enumerate(condition):
            if x == -1:
                # random condition (works currently only for binary conditions)
                x = np.random.randint(0, 2)
            cond_labels[n, i] = x

    # generate samples
    num_sequences = int(np.floor(num_samples_total / num_samples_parallel))
    all_samples = np.zeros((num_samples_parallel * num_sequences, sequence_length_total + n_conditions))
    print("Generating samples...")

    for i in range(num_sequences):
        print(f"Generating sequence {i+1} of {num_sequences}...")
        # init sequence for windows_slices
        sequence = torch.zeros((num_samples_parallel, seq_len_cond)).to(device)
        while sequence.shape[1] < sequence_length_total + seq_len_cond:
            # samples = gs.generate_samples(labels, num_samples=num_samples_parallel, conditions=True)
            z = Trainer.sample_latent_variable(batch_size=num_samples_parallel, latent_dim=latent_dim, device=device)
            z = torch.cat((z, cond_labels, sequence[:, -seq_len_cond:]), dim=1).type(torch.FloatTensor).to(device)
            samples = generator(z)
            sequence = torch.cat((sequence, samples.view(num_samples_parallel, -1)), dim=1)
        sequence = sequence[:, seq_len_cond:seq_len_cond+sequence_length_total]
        sequence = torch.cat((cond_labels, sequence), dim=1)
        all_samples[i * num_samples_parallel:(i + 1) * num_samples_parallel, :] = sequence.detach().cpu().numpy()

    # save samples
    print("Saving samples...")
    path = 'generated_samples'
    file = os.path.join(path, os.path.basename(file).split('.')[0] + '.csv')
    pd.DataFrame(all_samples).to_csv(file, index=False)

    print("Generated samples were saved to " + file)
