import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from helpers import system_inputs
from helpers.dataloader import Dataloader
from helpers.initialize_gan import init_gan, gan_types
from helpers.trainer import GANTrainer
from nn_architecture.models import DecoderGenerator, TransformerGenerator, AutoencoderGenerator
from nn_architecture.ae_networks import TransformerDoubleAutoencoder, TransformerAutoencoder, \
    TransformerFlattenAutoencoder

#another comment
def main():
    default_args = system_inputs.parse_arguments(sys.argv, file='generate_samples_main.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    num_samples_total = default_args['num_samples_total']
    num_samples_parallel = default_args['num_samples_parallel']
    kw_timestep_dataset = default_args['kw_timestep_dataset']
    average_over = default_args['average']

    condition = default_args['conditions']
    if not isinstance(condition, list):
        condition = [condition]
    # if no condition is given, make empty list
    if len(condition) == 1 and condition[0] == 'None':
        condition = []

    file = default_args['path_file']
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
        if not os.path.exists(path):
            os.makedirs(path)
        path_samples = os.path.join(path, path_samples)

    state_dict = torch.load(file, map_location='cpu')

    # load model/training configuration
    filename_dataset = state_dict['configuration']['path_dataset']
    n_conditions = state_dict['configuration']['n_conditions']
    n_channels = state_dict['configuration']['n_channels']
    channel_names = state_dict['configuration']['channel_names']
    latent_dim = state_dict['configuration']['latent_dim']
    sequence_length = state_dict['configuration']['sequence_length']
    input_sequence_length = state_dict['configuration']['input_sequence_length']

    assert n_conditions == len(condition), f"Number of conditions in model ({n_conditions}) does not match number of conditions given ({len(condition)})."

    if input_sequence_length != 0 and input_sequence_length != sequence_length:
        raise NotImplementedError(f"Prediction case detected.\nInput sequence length ({input_sequence_length}) > 0 and != sequence length ({sequence_length}).\nPrediction is not implemented yet.")

    # get data from dataset if sequence2sequence or prediction case
    if input_sequence_length != 0:
        dataloader = Dataloader(**state_dict['configuration']['dataloader'])
        dataset = dataloader.get_data()
        if n_conditions > 0:
            raise NotImplementedError(
                f"Prediction or Sequence-2-Sequence case detected.\nGeneration with conditions in on of these cases is not implemented yet.\nPlease generate without conditions.")
    else:
        dataset = None

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator
    print("Initializing generator...")
    latent_dim_in = latent_dim + n_conditions + n_channels if input_sequence_length > 0 else latent_dim + n_conditions

    for k, v in gan_types.items():
        if state_dict['configuration']['generator_class'] in v:
            gan_type = k
            print(gan_type)
            break

    generator, _ = init_gan(gan_type=gan_type,
                            latent_dim_in=latent_dim_in,
                            channel_in_disc=n_channels,
                            n_channels=n_channels,
                            n_conditions=n_conditions,
                            sequence_length_generated=sequence_length,
                            device=device,
                            hidden_dim=state_dict['configuration']['hidden_dim'],
                            num_layers=state_dict['configuration']['num_layers'],
                            activation=state_dict['configuration']['activation'],
                            input_sequence_length=input_sequence_length,
                            patch_size=state_dict['configuration']['patch_size'],
                            path_autoencoder=state_dict['configuration']['path_autoencoder'],
                            ae_sequence_length=sequence_length-state_dict['configuration']['padding'],
                            padding=state_dict['configuration']['padding'],
                            )
    generator.eval()
    if isinstance(generator, DecoderGenerator):
        generator.decode_output()

    # load generator weights
    generator.load_state_dict(state_dict['generator'])
    generator.to(device)

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
            raise ValueError(
                f"Number of conditions in model (={n_conditions}) does not match number of conditions given ={len(condition)}.")

    seq_len = max(1, input_sequence_length)
    cond_labels = torch.zeros((num_samples_parallel, seq_len, n_conditions)).to(device) + torch.tensor(condition).to(device)
    cond_labels = cond_labels.to(device)

    # generate samples
    num_sequences = num_samples_total // num_samples_parallel
    print("Generating samples...")

    all_samples = np.zeros((num_samples_parallel * num_sequences * n_channels, n_conditions + 1 + sequence_length))

    for i in range(num_sequences):
        print(f"Generating sequence {i + 1}/{num_sequences}...")
        # get input sequence by drawing randomly num_samples_parallel input sequences from dataset
        if input_sequence_length > 0 and dataset:
            input_sequence = dataset[np.random.randint(0, len(dataset), num_samples_parallel), :input_sequence_length, :]
            labels_in = torch.cat((cond_labels, input_sequence), dim=1).float()
        else:
            labels_in = cond_labels
            input_sequence = None
        with torch.no_grad():
            # draw latent variable
            z = GANTrainer.sample_latent_variable(batch_size=num_samples_parallel, latent_dim=latent_dim,
                                                  sequence_length=seq_len, device=device)
            # concat with conditions and input sequence
            z = torch.cat((z, labels_in), dim=-1).float().to(device)
            # generate samples
            print(seq_len)
            print(z.shape)
            
            samples = generator(z).cpu().numpy()
        # if prediction case, concatenate input sequence and generated sequence
        if input_sequence_length > 0 and input_sequence_length != sequence_length and input_sequence is not None:
            samples = np.concatenate((input_sequence, samples), axis=1)
        # reshape samples by concatenating over channels in incrementing channel name order
        new_samples = np.zeros((num_samples_parallel * n_channels, n_conditions + 1 + sequence_length))
        for j, channel in enumerate(channel_names):
            new_samples[j::n_channels] = np.concatenate((cond_labels.cpu().numpy()[:, 0, :], np.zeros((num_samples_parallel, 1)) + channel, samples[:, :, j]), axis=-1)
        # add samples to all_samples
        all_samples[i * num_samples_parallel * n_channels:(i + 1) * num_samples_parallel * n_channels] = new_samples

    # save samples
    print("Saving samples...")
    # check if column condition labels are given
    if state_dict['configuration']['dataloader']['column_label'] and len(
            state_dict['configuration']['dataloader']['column_label']) == n_conditions:
        col_labels = state_dict['configuration']['dataloader']['column_label']
    else:
        if n_conditions > 0:
            col_labels = [f'Condition {i}' for i in range(n_conditions)]
        else:
            col_labels = []
    # check if channel label is given
    if state_dict['configuration']['dataloader']['channel_label']:
        channel_label = [state_dict['configuration']['dataloader']['channel_label']]
    else:
        channel_label = ['Channel']
    # get keyword for time step labels
    if state_dict['configuration']['dataloader']['kw_timestep']:
        kw_timestep = state_dict['configuration']['dataloader']['kw_timestep']
    else:
        kw_timestep = 'Time'
    # create time step labels
    time_labels = [f'Time{i}' for i in range(sequence_length)]
    # create dataframe
    df = pd.DataFrame(all_samples, columns=[col_labels + channel_label + time_labels])
    df.to_csv(path_samples, index=False)

    print("Generated samples were saved to " + path_samples)


if __name__ == '__main__':
    # sys.argv = ["file=gan_1830ep.pt", "conditions=1"]
    main()
