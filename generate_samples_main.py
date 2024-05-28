import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader

from helpers import system_inputs
from helpers.dataloader import Dataloader
from helpers.initialize_gan import init_gan
from helpers.trainer import GANTrainer
from nn_architecture.models import DecoderGenerator
from nn_architecture.vae_networks import VariationalAutoencoder

#another comment
def main():
    default_args = system_inputs.parse_arguments(sys.argv, file='generate_samples_main.py')

    # set a seed for reproducibility if desired
    if default_args['seed'] is not None:
        np.random.seed(default_args['seed'])                       
        torch.manual_seed(default_args['seed'])                    
        torch.cuda.manual_seed(default_args['seed'])               
        torch.cuda.manual_seed_all(default_args['seed'])           
        torch.backends.cudnn.deterministic = True  
    
    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    num_samples_total = default_args['num_samples_total']
    num_samples_parallel = default_args['num_samples_parallel']

    condition = default_args['conditions']
    if not isinstance(condition, list):
        condition = [condition]
    # if no condition is given, make empty list
    if len(condition) == 1 and condition[0] == 'None':
        condition = []

    file = default_args['model']
    if file.split(os.path.sep)[0] == file and file.split('/')[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    path_samples = default_args['save_name']
    if path_samples == '':
        # Use checkpoint filename as path
        path_samples = os.path.basename(file).split('.')[0] + '.csv'
    if path_samples.split(os.path.sep)[0] == path_samples:
        # use default path if no path is given
        path = 'generated_samples'
        if not os.path.exists(path):
            os.makedirs(path)
        path_samples = os.path.join(path, path_samples)

    state_dict = torch.load(file, map_location='cpu')

    # define device
    device = torch.device('cpu')

    # check if column condition labels are given
    n_conditions = len(state_dict['configuration']['kw_conditions']) if state_dict['configuration']['kw_conditions'] and state_dict['configuration']['kw_conditions'] != [''] else 0
    if n_conditions > 0:        
        col_labels = state_dict['configuration']['dataloader']['kw_conditions']
    else:
        col_labels = []
            
    # check if channel label is given
    if not state_dict['configuration']['dataloader']['kw_channel'] in [None, '']:
        kw_channel = [state_dict['configuration']['dataloader']['kw_channel']]
    else:
        kw_channel = ['Electrode']

    # get keyword for time step labels
    if state_dict['configuration']['dataloader']['kw_time']:
        kw_time = state_dict['configuration']['dataloader']['kw_time']
    else:
        kw_time = 'Time'

    if state_dict['configuration']['model_class'] != 'VariationalAutoencoder':

        # load model/training configuration
        n_conditions = state_dict['configuration']['n_conditions']
        n_channels = state_dict['configuration']['n_channels']
        channel_names = state_dict['configuration']['channel_names']
        latent_dim = state_dict['configuration']['latent_dim']
        sequence_length = state_dict['configuration']['sequence_length']
        # input_sequence_length = state_dict['configuration']['input_sequence_length']

        if n_conditions != len(condition):
            raise ValueError(f"Number of conditions in model ({n_conditions}) does not match number of conditions given ({len(condition)}).")

        # if input_sequence_length != 0 and input_sequence_length != sequence_length:
        #     raise NotImplementedError(f"Prediction case detected.\nInput sequence length ({input_sequence_length}) > 0 and != sequence length ({sequence_length}).\nPrediction is not implemented yet.")

        # get data from dataset if sequence2sequence or prediction case
        # if input_sequence_length != 0:
        #     dataloader = Dataloader(**state_dict['configuration']['dataloader'])
        #     dataset = dataloader.get_data()
        #     if n_conditions > 0:
        #         raise NotImplementedError(
        #             f"Prediction or Sequence-2-Sequence case detected.\nGeneration with conditions in on of these cases is not implemented yet.\nPlease generate without conditions.")
        # else:
        #     dataset = None

        # define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generator
        print("Initializing generator...")
        # latent_dim_in = latent_dim + n_conditions + n_channels if input_sequence_length > 0 else latent_dim + n_conditions
        latent_dim_in = latent_dim + n_conditions

        generator, _ = init_gan(latent_dim_in=latent_dim_in,
                                channel_in_disc=n_channels,
                                n_channels=n_channels,
                                n_conditions=n_conditions,
                                sequence_length_generated=sequence_length,
                                device=device,
                                hidden_dim=state_dict['configuration']['hidden_dim'],
                                num_layers=state_dict['configuration']['num_layers'],
                                # activation=state_dict['configuration']['activation'],
                                # input_sequence_length=input_sequence_length,
                                patch_size=state_dict['configuration']['patch_size'],
                                autoencoder=state_dict['configuration']['autoencoder'],
                                # padding=state_dict['configuration']['padding'],
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

        seq_len = 1  # max(1, input_sequence_length)
        cond_labels = torch.zeros((num_samples_parallel, seq_len, n_conditions)).to(device) + torch.tensor(condition).to(device)
        cond_labels = cond_labels.to(device)

        # generate samples
        num_sequences = num_samples_total // num_samples_parallel
        print("Generating samples...")

        all_samples = np.zeros((num_samples_parallel * num_sequences * n_channels, n_conditions + 1 + sequence_length))

        for i in range(num_sequences):
            print(f"Generating sequence {i + 1}/{num_sequences}...")
            # get input sequence by drawing randomly num_samples_parallel input sequences from dataset
            # if input_sequence_length > 0 and dataset:
            #     input_sequence = dataset[np.random.randint(0, len(dataset), num_samples_parallel), :input_sequence_length, :]
            #     labels_in = torch.cat((cond_labels, input_sequence), dim=1).float()
            # else:
            # input_sequence = None
            # labels_in = cond_labels
            with torch.no_grad():
                # draw latent variable
                z = GANTrainer.sample_latent_variable(batch_size=num_samples_parallel, latent_dim=latent_dim,
                                                    sequence_length=seq_len, device=device)
                # concat with conditions and input sequence
                z = torch.cat((z, cond_labels), dim=-1).float().to(device)
                # generate samples
                samples = generator(z).cpu().numpy()
            # if prediction case, concatenate input sequence and generated sequence
            # if input_sequence_length > 0 and input_sequence_length != sequence_length and input_sequence is not None:
            #     samples = np.concatenate((input_sequence, samples), axis=1)
            # reshape samples by concatenating over channels in incrementing channel name order
            new_samples = np.zeros((num_samples_parallel * n_channels, n_conditions + 1 + sequence_length))
            for j, channel in enumerate(channel_names):
                # padding = np.zeros((samples.shape[0], state_dict['configuration']['padding']))
                # new_samples[j::n_channels] = np.concatenate((cond_labels.cpu().numpy()[:, 0, :], np.zeros((num_samples_parallel, 1)) + channel, np.concatenate((samples[:, :, j], padding), axis=1)), axis=-1)
                new_samples[j::n_channels] = np.concatenate((cond_labels.cpu().numpy()[:, 0, :], np.zeros((num_samples_parallel, 1)) + channel, samples[:, :, j]), axis=-1)
            # add samples to all_samples
            all_samples[i * num_samples_parallel * n_channels:(i + 1) * num_samples_parallel * n_channels] = new_samples

    elif state_dict['configuration']['model_class'] == 'VariationalAutoencoder':

        # load data
        dataloader = Dataloader(path=state_dict['configuration']['dataloader']['data'],
                        kw_channel=kw_channel[0], 
                        kw_conditions=state_dict['configuration']['dataloader']['kw_conditions'],
                        kw_time=state_dict['configuration']['dataloader']['kw_time'],
                        norm_data=state_dict['configuration']['dataloader']['norm_data'], 
                        std_data=state_dict['configuration']['dataloader']['std_data'], 
                        diff_data=state_dict['configuration']['dataloader']['diff_data'])        
        dataset = dataloader.get_data()
        dataset = DataLoader(dataset, batch_size=state_dict['configuration']['batch_size'], shuffle=True)

        sequence_length = int(state_dict['configuration']['input_dim']/dataset.dataset.shape[-1])
        channel_names = dataloader.channels
        n_conditions = len(default_args['conditions'])
        if condition:
            cond_labels = torch.zeros((num_samples_total, state_dict['configuration']['input_dim'], len(default_args['conditions']))).to(device) + torch.tensor(condition).to(device)
        else:
            cond_labels = torch.zeros((num_samples_total, state_dict['configuration']['input_dim'], 1)).to(device) + torch.tensor([-1]).to(device)
        cond_labels = cond_labels.to(device)

        # load VAE
        model = VariationalAutoencoder(input_dim=state_dict['configuration']['input_dim'], 
                                   hidden_dim=state_dict['configuration']['hidden_dim'], 
                                   encoded_dim=state_dict['configuration']['encoded_dim'], 
                                   activation=state_dict['configuration']['activation'],
                                   device=device).to(device)
        
        consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')
        model.load_state_dict(state_dict['model'])

        # generate samples
        samples = model.generate_samples(loader=dataset, condition=condition, num_samples=num_samples_total)

        # reconfigure samples to a 2D matrix for saving
        new_samples = []
        for j, channel in enumerate(channel_names):
            new_samples.append(np.concatenate((cond_labels.cpu().numpy()[:, 0, :], np.zeros((num_samples_total, 1)) + channel, samples[:, 1:, j]), axis=-1))
        # add samples to all_samples
        all_samples = np.vstack(new_samples)
    
    else:
        raise NotImplementedError(f"The model class {state_dict['configuration']['model_class']} is not recognized.")

    # save samples
    print("Saving samples...")

    # create time step labels
    time_labels = [f'{kw_time}{i}' for i in range(sequence_length)]
    # create dataframe
    df = pd.DataFrame(all_samples, columns=[col_labels + kw_channel + time_labels])
    df.to_csv(path_samples, index=False)

    print("Generated samples were saved to " + path_samples)
        
if __name__ == '__main__':
    # sys.argv = ["file=gan_1830ep.pt", "conditions=1"]
    main()
