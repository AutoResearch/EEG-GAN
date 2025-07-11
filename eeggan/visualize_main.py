import sys
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from eeggan.helpers import system_inputs
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.visualize_pca import visualization_dim_reduction
from eeggan.helpers.visualize_spectogram import plot_fft_hist, plot_spectogram


def main(args=None):
    #Determine args
    if args is None:
        default_args = system_inputs.parse_arguments(sys.argv, file='visualize_main.py')
    else:
        default_args = system_inputs.parse_arguments(args, file='visualize_main.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')
    
    if default_args['data'] != '' and default_args['model'] != '':
        raise ValueError("Please specify only one of the following arguments: data, model")

    if default_args['channel_index'][0] > -1 and (default_args['pca'] or default_args['tsne']):
        print("Warning: channel_index is set to a specific value, but PCA or t-SNE is enabled.\n"
              "PCA and t-SNE are only available for all channels. Ignoring channel_index.")

    # throw error if checkpoint but csv-file is specified
    if default_args['model'] != '' and not default_args['model'].endswith('.pt'):
        raise ValueError("Inconsistent parameter specification. 'model' was specified but no model-file (.pt) was given.")
    if default_args['data'] != '' and not default_args['data'].endswith('.csv'):
        raise ValueError("Inconsistent parameter specification. 'data' was specified but no csv-file was given.")

    # throw warning if checkpoint and conditions are given
    if default_args['model'] != '' and default_args['kw_conditions'][0] != '':
        warnings.warn("Conditions are given, but model is specified. Given conditions will be ignored and taken from the model file if the model file contains the conditions parameter.")

    original_data = None
    if default_args['data'] != '':
        n_conditions = len(default_args['kw_conditions']) if default_args['kw_conditions'][0] != '' else 0
        # load data with DataLoader
        dataloader = Dataloader(path=default_args['data'],
                                norm_data=True,
                                kw_time=default_args['kw_time'],
                                kw_conditions=default_args['kw_conditions'],
                                kw_channel=default_args['kw_channel'],)
        data = dataloader.get_data(shuffle=False)[:, n_conditions:].numpy()
        conditions = dataloader.get_labels()[:, :, 0].numpy()
        random = True
    elif default_args['model'] != '':
        state_dict = torch.load(default_args['model'], map_location='cpu')
        n_conditions = state_dict['configuration']['n_conditions'] if 'n_conditions' in state_dict['configuration'].keys() else 0
        data = np.concatenate(state_dict['samples'])
        if len(data.shape) == 2:
            data = data.reshape((1, data.shape[0], data.shape[1]))
        if len(data.shape) == 3:
            conditions = data[:, :n_conditions, 0]
            data = data[:, n_conditions:]
        elif len(data.shape) == 4:
            # autoencoder samples are saved as (n_samples, type, sequence_length, n_channels)
            # type = 0: original, type = 1: reconstructed
            conditions = data[:, 0, :n_conditions, 0]
            original_data = data[:, 0, n_conditions:]
            data = data[:, 1, n_conditions:]

            # set channel_plots to True if original_data was found in samples
            if not default_args['channel_plots'] and data.shape[-1] > 1:
                default_args['channel_plots'] = True
                warnings.warn("Original data was found in checkpoint and data contains more than 1 channel. Setting channel_plots to True to improve the visualization quality.")
        else:
            raise ValueError(f"Invalid shape of data: {data.shape}")
        random = False
    else:
        raise ValueError("Please specify one of the following arguments: csv, checkpoint")

    # set channel index
    if default_args['channel_index'][0] == -1:
        channel_index = np.arange(data.shape[-1])
    else:
        channel_index = default_args['channel_index']

    # -----------------------------
    # Normal curve plotting
    # -----------------------------

    if default_args['n_samples'] > 0:
        print(f"Plotting {default_args['n_samples']} samples...")

        # create a normal curve plot
        if default_args['n_samples'] > data.shape[0]:
            warnings.warn(f"n_samples ({default_args['n_samples']}) is larger than the number of samples ({data.shape[0]}).\n"
                          f"Plotting all available samples instead.")
            default_args['n_samples'] = data.shape[0]

        if random:
            index = np.random.randint(0, data.shape[0]-1, default_args['n_samples'])
        else:
            index = np.linspace(0, data.shape[0]-1, default_args['n_samples'], dtype=int)

        ncols = 1 if not default_args['channel_plots'] else len(channel_index)
        fig, axs = plt.subplots(nrows=default_args['n_samples'], ncols=ncols)
        picking_type = 'randomly' if random else 'evenly'
        if original_data is not None:
            comparison = '; reconstructed (blue) vs original (orange)'
        else:
            comparison = ''
        fig.suptitle(f'{picking_type} picked samples' + comparison)

        for irow, i in enumerate(index):
            if ncols == 1:
                for j in channel_index:
                    if default_args['n_samples'] == 1:
                        axs.plot(data[i, :, j])
                        if original_data is not None:
                            axs.plot(original_data[i, :, j])
                    else:
                        axs[irow].plot(data[i, :, j])
                        if original_data is not None:
                            axs[irow].plot(original_data[i, :, j])
            else:
                for jcol, j in enumerate(channel_index):
                    if default_args['n_samples'] == 1:
                        axs[jcol].plot(data[i, :, j])
                        if original_data is not None:
                            axs[jcol].plot(original_data[i, :, j])
                    else:
                        axs[irow, jcol].plot(data[i, :, j])
                        if original_data is not None:
                            axs[irow, jcol].plot(original_data[i, :, j])

        plt.show()

    # -----------------------------
    # Loss plotting
    # -----------------------------

    try:
        if default_args['loss'] and default_args['model'] == '':
            raise ValueError("Loss plotting only available for checkpoint and not csv")
        elif default_args['loss']:
            print("Plotting losses...")
            # get all losses from state_dict
            for key in state_dict.keys():
                if 'loss' in key:
                    plt.plot(state_dict[key], label=key, marker='.')
            plt.title('training losses')
            plt.legend()
            plt.show()
    except ValueError as e:
        print(e)
        
    # -----------------------------
    # Average plotting
    # -----------------------------

    if default_args['average']:
        if n_conditions == 0:
            print("Plotting averaged curves...")
        else:
            print("Plotting averaged curves over each set of conditions...")
        # average over conditions
        if n_conditions > 0:
            conditions_set = np.unique(conditions, axis=0)
            # sort samples by condition sets
            averaged_data = []
            for i, cond in enumerate(conditions_set):
                index_cond = np.where(np.sum(conditions == cond, axis=1) == n_conditions)
                averaged_data.append(np.mean(data[index_cond], axis=0))
            # average over samples
            averaged_data = np.array(averaged_data)
        else:
            averaged_data = np.mean(data, axis=0).reshape(1, data.shape[1], data.shape[2])
            conditions_set = ['']

        # plot averaged data
        ncols = 1 if not default_args['channel_plots'] else len(channel_index)
        nrows = averaged_data.shape[0]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
        if n_conditions == 0:
            fig.suptitle('averaged curves')
        else:
            fig.suptitle('averaged curves over conditions')
        for i, cond in enumerate(conditions_set):
            if ncols == 1:
                for j in channel_index:
                    if nrows == 1:
                        axs.plot(averaged_data[i, :, j])
                    else:
                        axs[i].plot(averaged_data[i, :, j])
            else:
                for jcol, j in enumerate(channel_index):
                    if nrows == 1:
                        axs[jcol].plot(averaged_data[i, :, j])
                    else:
                        axs[i, jcol].plot(averaged_data[i, :, j])
             # set legend at the right hand side of the plot;
            # legend carries the condition information
            # make graph and legend visible within the figure
            if not default_args['channel_plots']:
                if nrows == 1:
                    axs.legend([f'{cond}'], loc='center right', bbox_to_anchor=(1, 0.5))
                else:
                    axs[i].legend([f'{cond}'], loc='center right', bbox_to_anchor=(1, 0.5))
            else:
                if nrows == 1:
                    axs[-1].legend([f'{cond}'], loc='center right', bbox_to_anchor=(1, 0.5))
                else:
                    axs[i, -1].legend([f'{cond}'], loc='center right', bbox_to_anchor=(1, 0.5))
        plt.show()

    # -----------------------------
    # PCA and t-SNE plotting
    # -----------------------------

    if default_args['pca'] or default_args['tsne']:
        if original_data is None and default_args['comp_data'] != '':
            # load comparison data
            dataloader_comp = Dataloader(path=default_args['comp_data'],
                                         norm_data=True,
                                         kw_time=default_args['kw_time'],
                                         kw_conditions=default_args['kw_conditions'],
                                         kw_channel=default_args['kw_channel'], )
            original_data = dataloader_comp.get_data(shuffle=False)[:, n_conditions:].numpy()
        elif original_data is None and default_args['comp_data'] == '':
            raise ValueError("No comparison data found for PCA or t-SNE. Please specify a comparison dataset with the argument 'comp_data'.")

        if default_args['pca']:
            print("Plotting PCA...")
            visualization_dim_reduction(original_data, data, 'pca', False, 'pca_file')

        if default_args['tsne']:
            print("Plotting t-SNE...")
            visualization_dim_reduction(original_data, data, 'tsne', False, 'tsne_file')

    # -----------------------------
    # Spectogram plotting
    # -----------------------------

    if default_args['spectogram']:
        print("Plotting spectograms...")
        if data.shape[-1] > 1:
            warnings.warn(f"Spectogram plotting is only available for 1 channel but {data.shape[-1]} channels were given. Plotting only the first channel instead.")
            fft_data = data[:, :, 0]
        else:
            fft_data = data
        plot_spectogram(fft_data)

    # -----------------------------
    # FFT plotting
    # -----------------------------

    if default_args['fft']:
        print("Plotting FFT...")
        if data.shape[-1] > 1:
            warnings.warn(f"FFT plotting is only available for 1 channel but {data.shape[-1]} channels were given. Plotting only the first channel instead.")
            fft_data = data[:, :, 0]
        else:
            fft_data = data
        plot_fft_hist(fft_data)


if __name__ == '__main__':
    main()