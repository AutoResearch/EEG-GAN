import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from helpers import system_inputs
from helpers.dataloader import Dataloader
from helpers.visualize_pca import visualization_dim_reduction
from helpers.visualize_spectogram import plot_fft_hist, plot_spectogram


class Plotter:
    """Plotting class"""

    CHECKPOINT = 2
    EXPERIMENT = 3
    CSV = 4

    def __init__(self):
        pass

    def plot(self, title, xlabel, ylabel, n_subplots=8, n_plots=1, save_path=None):
        pass


def main():
    default_args = system_inputs.parse_arguments(sys.argv, file='visualize_main.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    if default_args['csv'] + default_args['checkpoint'] != 1:  #  + default_args['experiment']
        raise ValueError("Please specify only one of the following arguments: csv, checkpoint")

    if default_args['channel_index'][0] > -1 and (default_args['pca'] or default_args['tsne']):
        print("Warning: channel_index is set to a specific value, but PCA or t-SNE is enabled.\n"
              "PCA and t-SNE are only available for all channels. Ignoring channel_index.")

    # throw error if checkpoint but csv-file is specified
    if default_args['checkpoint'] and default_args['path_dataset'].split('.')[-1] == 'csv':
        raise ValueError("Inconsistent parameter specification. 'checkpoint' was specified but a csv-file was given.")

    # throw warning if checkpoint and conditions are given
    if default_args['checkpoint'] and default_args['conditions'][0] != '':
        warnings.warn("Conditions are given, but checkpoint is specified. Given conditions are ignored since they will be taken directly from the checkpoint file.")

    # throw error if PCA is enabled but no comparison dataset is given
    if default_args['pca'] and default_args['path_comp_dataset'] == '':
        raise ValueError("PCA computation is True but keyword 'path_comp_dataset' is not given. Please specify a path to a dataset for comparison of the PCA.")

    # throw error if t-SNE is enabled but no comparison dataset is given
    if default_args['tsne'] and default_args['path_comp_dataset'] == '':
        raise ValueError("t-SNE computation is True but keyword 'path_comp_dataset' is not given. Please specify a path to a dataset for comparison of the t-SNE.")

    if default_args['csv']:
        n_conditions = len(default_args['conditions']) if default_args['conditions'][0] != '' else 0
        # load data with DataLoader
        dataloader = Dataloader(path=default_args['path_dataset'],
                                norm_data=True,
                                kw_timestep=default_args['kw_timestep'],
                                col_label=default_args['conditions'],
                                channel_label=default_args['channel_label'], )
        data = dataloader.get_data(shuffle=False)[:, n_conditions:].numpy()
        conditions = dataloader.get_labels()[:, :, 0].numpy()
        random = True
    elif default_args['checkpoint']:
        state_dict = torch.load(default_args['path_dataset'], map_location='cpu')
        n_conditions = state_dict['configuration']['n_conditions']
        sequence_length_generated = state_dict['configuration']['sequence_length_generated']
        data = np.stack(state_dict['generated_samples'])
        conditions = data[:, :n_conditions, 0]
        data = data[:, n_conditions:]
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
            index = np.random.randint(0, data.shape[0], default_args['n_samples'])
        else:
            index = np.linspace(0, data.shape[0], default_args['n_samples'], dtype=int)

        ncols = 1 if not default_args['channel_plots'] else len(channel_index)
        fig, axs = plt.subplots(nrows=default_args['n_samples'], ncols=ncols)
        picking_type = 'randomly' if random else 'evenly'
        fig.suptitle(f'{picking_type} picked samples')

        for irow, i in enumerate(index):
            if ncols == 1:
                for j in channel_index:
                    if default_args['n_samples'] == 1:
                        axs.plot(data[i, :, j])
                    else:
                        axs[irow].plot(data[i, :, j])
            else:
                for jcol, j in enumerate(channel_index):
                    if default_args['n_samples'] == 1:
                        axs[jcol].plot(data[i, :, j])
                    else:
                        axs[irow, jcol].plot(data[i, :, j])

        plt.show()

    # -----------------------------
    # Loss plotting
    # -----------------------------

    try:
        if default_args['loss'] and not default_args['checkpoint']:
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
            # axs[i].set_title(f'condition {cond}')
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
        # load comparison data
        dataloader_comp = Dataloader(path=default_args['path_comp_dataset'],
                                     norm_data=True,
                                     kw_timestep=default_args['kw_timestep'],
                                     col_label=default_args['conditions'],
                                     channel_label=default_args['channel_label'], )
        data_comp = dataloader_comp.get_data(shuffle=False)[:, n_conditions:].numpy()

        if default_args['pca']:
            visualization_dim_reduction(data_comp, data, 'pca', False, 'pca_file')

        if default_args['tsne']:
            visualization_dim_reduction(data_comp, data, 'tsne', False, 'tsne_file')

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
    # sys.argv = [
    #             'csv',
    #             'path_dataset=../generated_samples/gan_1ep_2chan_1cond.csv',
    #             # 'checkpoint',
    #             # 'path_dataset=../trained_models/gan_1ep_2chan_1cond.pt',
    #             'conditions=Condition',
    #             'channel_label=Electrode',
    #             'n_samples=8',
    #             # 'channel_plots',
    #             'channel_index=0',
    #             # 'loss',
    #             # 'average',
    #             'spectogram',
    #             'fft',
    #             # 'pca',
    #             # 'tsne',
    #             # 'path_comp_dataset=../data/gansMultiCondition_SHORT.csv',
    #             'path_comp_dataset=../data/gansMultiCondition.csv',
    # ]
    main()