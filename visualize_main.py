import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import system_inputs
import models
from dataloader import Dataloader
from generate_samples_main import GenerateSamples


class PlotterGanTraining:
    """This class is used to read samples from a csv-file and plot them.
    Shape of the csv-file has to be (rows: samples, cols: (conditions, signal))"""

    def __init__(self, load_file=True, filename=None, gan_or_emb='gan',
                 get_original_dataset=False, n_conditions=1):
        """if timestamp is None read files from directory generated_samples and
        use most recent file indicated by the timestamp.
        :param load_file: Boolean; if True load file to obtain dataset with time series samples.
        :param file: String; If a specific file is to be read, the filename can be specified here.
        :param gan_or_emb: String; 'gan' or 'emb' to indicate whether the file is a GAN or embedding sample.
        :param load_data: Boolean; std, mean, min and max are computed with the original dataset.
        :param n_conditions: Integer; Number of columns with conditions in the dataset before measurement."""

        if gan_or_emb == 'gan':
            self.filename = 'sample'
            self.filetype = 'gan'
        elif gan_or_emb == 'emb':
            self.filename = 'embedding'
            self.filetype = 'emb'
        else:
            raise ValueError('gan_or_emb has to be either gan or emb')

        self.n_conditions = n_conditions
        self.title = None
        self.ylim = None

        self.file = filename
        if load_file:
            if filename is None:
                self.file = self.read_files()[-1]  # Store most recent file indicated by timestamp
            else:
                # Check if file is a csv file
                if not filename.endswith('.csv'):
                    raise ValueError('File is not a csv-file. Please choose a csv-file')
                self.file = filename

            self.df = self.read_file()

            print("File: " + self.file)
        # else:
        #     print("No file loaded. Please use set_dataset to set the dataset.")

        self.dataloader = None
        if get_original_dataset:
            # Instantiate Dataloader
            path = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganAverageERP.csv'
            self.dataloader = Dataloader(path, diff_data=False, std_data=False, norm_data=True)

    def read_files(self):
        """This function reads the files from the directory generated_samples and returns a list of the files"""
        files = []
        for file in os.listdir('./generated_samples'):
            if file.endswith('.csv'):
                files.append(file)
        files = [s for s in files if self.filename in s]
        files.sort()
        return files

    def read_file(self):
        """This function reads the file and returns a pandas dataframe"""
        if self.filetype == 'gan':
            return pd.read_csv(os.path.join(r'.\generated_samples', self.file), header=None)
        else:
            return pd.read_csv(os.path.join(r'.\generated_samples', self.file), header=None).T

    def plot(self, stacked=False, batch_size=None, rows=None, n_samples=None, save=False):
        """
        This function plots the generated samples
        :param stacked: (Bool) Defines if the samples are plotted in a stacked manner (incremented value range)
        :param batch_size: (Int) Defines the batch size for one plot (number of plottet samples in one plot)
        :param rows: (Int) Defines the number of the starting row of the sample-file; negative numbers are allowed
        :param n_samples: (Int) Defines the number of samples to be plotted; samples are drawn uniformly from the file
        :return:
        """

        # get std and mean of the original dataset
        if self.dataloader is not None:
            # mean = self.dataloader.get_mean().detach().cpu().numpy()
            # std = self.dataloader.get_std().detach().cpu().numpy()
            mean, std = 0, 1
            data_min = self.dataloader.dataset_min.detach().cpu().numpy()
            data_max = self.dataloader.dataset_max.detach().cpu().numpy()
        else:
            mean = 0
            std = 1
            data_min = 0
            data_max = 1

        # re-transform data
        if isinstance(self.df, pd.DataFrame):
            # kick out row with column names and convert to 2D numpy array
            self.df = self.df.to_numpy()[1:, :]
        df = self.df[:, self.n_conditions:]
        # check if df has non-numeric values in the first row
        if isinstance(df[0, 0], str):
            df = df[1:, :].astype(float)
        df = (df * std - mean) * (data_max - data_min) + data_min

        if self.filetype == 'emb':
            df = df.T
            if np.abs(rows*2) < df.shape[0]:
                rows *= 2
            else:
                rows = -df.shape[0] if rows < 0 else df.shape[0]
            if batch_size is not None:
                batch_size *= 2 if np.abs(batch_size)*2 < df.shape[0] else df.shape[0]

        # Determine the starting row from which the samples are drawn
        if rows is None:
            rows = 0
        if rows < 0:
            rows = df.shape[0] + rows
        if self.filetype == 'emb' and rows % 2 != 0:
            Warning('The starting row has to be even for the embedding file. '
                    'The starting row is set to the next even number.')
            rows += 1
        if rows > 0:
            df = df[rows:, :]
            remaining_samples = df.shape[0] if self.filetype == 'gan' else int(df.shape[0]/2)
            print(f"number of remaining samples after cutting off the first rows: {remaining_samples}")

        # Determine the number of samples to be plotted
        sampling = True
        if n_samples is None:
            sampling = False
            if self.filetype == 'gan':
                n_samples = df.shape[0]
            else:
                n_samples = int(df.shape[0]/2)
        if (n_samples > df.shape[0] and self.filetype == 'gan') or (n_samples > df.shape[0] / 2 and self.filetype == 'emb'):
            Warning('n_samples is larger than the number of samples in the file. All samples are plotted.')
            if self.filetype == 'gan':
                n_samples = df.shape[0]
            else:
                n_samples = int(df.shape[0]/2)

        # Draw samples uniformly from the file according to n_samples
        index = np.linspace(0, df.shape[0] - 2, n_samples).astype(int)
        if self.filetype == 'emb':
            index_real = index
            # all indexes have to be uneven in the case of embedding files
            for i in range(n_samples):
                if index_real[i] % 2 != 0:
                    index_real[i] += 1
            index_rec = index_real + 1
            index = [i for j in zip(index_real, index_rec) for i in j]
        df = df[index, :]

        # If batch_size is None plot all samples in one plot
        # Else plot the samples in batches of batch_size
        if batch_size is None:
            batch_size = df.shape[0]
        else:
            if (batch_size > n_samples and self.filetype == 'gan') or (batch_size > n_samples/2 and self.filetype == 'emb'):
                batch_size = n_samples
                Warning('batch_size is bigger than n_samples. Setting batch_size = n_samples')

        for i in range(0, df.shape[0], batch_size):
            batch = df[i:i + batch_size]
            if stacked:
                # Plot the continuous signals in subplots
                fig, axs = plt.subplots(batch_size, sharex='all', sharey='all')
                if self.filetype == 'gan':
                    for j in range(0, batch.shape[0]):
                        # plot each generated sample in another subplot
                        axs[batch_size-1-j].plot(batch[j], 'c')
                        if sampling:
                            axs[batch_size-1-j].text(df.shape[1], batch[j].mean(), f'sample: {rows+index[i+j]}', horizontalalignment='right')
                elif self.filetype == 'emb':
                    # plot the real and the reconstructed sample
                    for j in range(0, batch.shape[0], 2):
                        axs[batch_size-1-j].plot(batch[j], 'c')
                        axs[batch_size-1-j].plot(batch[j + 1], 'orange')
                        if sampling:
                            axs[batch_size-1-j].text(df.shape[1], batch[j].mean(), f'sample: {rows+index[i+j]}', horizontalalignment='right')
                        plt.legend(['real', 'reconstructed'])
                if self.title is not None:
                    axs[0].set_title(self.title)
            else:
                # plot all samples in one plot
                plt.plot(df)
                if self.title is not None:
                    plt.title(self.title)

            # if self.ylim is not None:
            #     plt.ylim(self.ylim)

            if save:
                path = 'plots'
                if not os.path.exists(path):
                    os.makedirs(path)
                file = self.file.split(os.path.sep)[-1].split('.')[0]
                plt.savefig(os.path.join(path, f'{file}_{i}.png'), dpi=600)
            else:
                plt.show()

    def set_dataset(self, gen_samples):
        """Set the dataset with the generated samples.
        :param gen_samples: Tensor or Numpy.array or pd.DataFrame; Generated samples.
                            Shape: (rows: samples, cols: (conditions, signal))"""
        if isinstance(gen_samples, torch.Tensor):
            gen_samples = gen_samples.detach().cpu().numpy()

        if len(gen_samples.shape) > 2:
            gen_samples = gen_samples.reshape(-1, gen_samples.shape[-1])

        if not isinstance(gen_samples, pd.DataFrame):
            self.df = pd.DataFrame(gen_samples, columns=np.arange(gen_samples.shape[1]), index=np.arange(gen_samples.shape[0]))
        else:
            self.df = gen_samples

    def get_dataset(self, column=False, index=False, n_conditions=0):
        dataset = self.df.to_numpy()
        # if not column:
        #     dataset = dataset[1:, :]
        # if not index:
        #     dataset = dataset[:, 1:]
        # if n_conditions > 0:
        #     dataset = dataset[:, n_conditions:]
        return dataset

    def set_title(self, title):
        self.title = title

    def set_y_lim(self, max, min=None):
        """Set the y-axis limits.
        :param max: integer or tuple of integers; Maximum value of the y-axis. If tuple ylim = max.
        :param min: integer; Minimum value of the y-axis. If None ylim = (0, max)."""

        if isinstance(max, tuple) and len(max) == 2 and min is None:
            self.ylim = max
        elif isinstance(max, int) and isinstance(min, int):
            self.ylim = (min, max)
        elif (isinstance(max, tuple) and len(max) == 1) and min is None:
            self.ylim = (0, max[0])
        elif isinstance(max, int) and min is None:
            self.ylim = (0, max)
        elif isinstance(max, tuple) and isinstance(min, int):
            raise ValueError('max cannot be a 2D tuple if min is an int')
        elif max is None:
            raise ValueError('max cannot be None')

        if not isinstance(self.ylim[0], int) or not isinstance(self.ylim[1], int):
            raise ValueError('max and min have to be integers')

        if self.ylim[0] > self.ylim[1]:
            raise ValueError('min cannot be bigger than max')

        if self.ylim[0] == self.ylim[1]:
            raise ValueError('min cannot be equal to max')

        self.ylim = (min, max)


if __name__ == '__main__':

    # get default system arguments
    system_args = system_inputs.default_inputs_visualize()
    default_args = {}
    for key, value in system_args.items():
        # value = [type, description, default value]
        default_args[key] = value[2]

    # Get system arguments
    file, generate, experiment, checkpoint, csv_file, plot_losses, save, \
        n_samples, batch_size, starting_row, n_conditions, \
        bandpass, mvg_avg, mvg_avg_window \
        = None, None, None, None, None, None, None, None, None, None, None, None, None, None

    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')
    for arg in sys.argv:
        if '.py' not in arg:
            if arg == 'help':
                helper = system_inputs.HelperVisualize('visualize_main.py', system_inputs.default_inputs_visualize())
                helper.print_table()
                helper.print_help()
                exit()
            elif arg == 'generate':
                print('Using generator to create samples')
                generate = True
            elif arg == 'experiment':
                print('Drawing samples from experiment dataset')
                experiment = True
            elif arg == 'checkpoint':
                print('Drawing samples from a training checkpoint file')
                checkpoint = True
            elif arg == 'csv_file':
                print('Drawing samples from a csv-file')
                csv_file = True
            elif arg == 'bandpass':
                print('Using bandpass filter on samples')
                bandpass =True
            elif arg == 'mvg_avg':
                print('Using moving average filter on samples')
                mvg_avg = True
            elif arg == 'plot_losses':
                print('Plotting losses')
                plot_losses = True
            elif arg == 'save':
                print('Saving plots to directory "plots"')
                save = True
            elif '=' in arg:
                kw = arg.split('=')
                if kw[0] == 'file':
                    print(f'Using file: {kw[1]}')
                    file = kw[1]
                elif kw[0] == 'n_conditions':
                    print(f'Number of conditions: {kw[1]}')
                    n_conditions = int(kw[1])
                elif kw[0] == 'n_samples':
                    print(f'Number of samples: {kw[1]}')
                    n_samples = int(kw[1])
                elif kw[0] == 'batch_size':
                    print(f'Batch size: {kw[1]}')
                    batch_size = int(kw[1])
                elif kw[0] == 'starting_row':
                    print(f'Start to draw samples from row: {kw[1]}')
                    rows = int(kw[1])
                elif kw[0] == 'mvg_avg_window':
                    print(f'Window of moving average filter: {kw[1]}')
                    mvg_avg_window = int(kw[1])
                else:
                    print(f'Keyword {kw[0]} not recognized. Please use the keyword "help" to see the available arguments.')
            else:
                print(f'Keyword {arg} not recognized. Please use the keyword "help" to see the available arguments.')

    # ----------------------------
    # configuration of program
    # ----------------------------

    generate = default_args['generate'] if generate is None else generate
    experiment = default_args['experiment'] if experiment is None else experiment
    checkpoint = default_args['checkpoint'] if checkpoint is None else checkpoint
    csv_file = default_args['csv_file'] if csv_file is None else csv_file

    if sum([generate, experiment, checkpoint, csv_file]) > 1:
        raise ValueError('Only one of the following options can be active: generate, experiment, checkpoint, csv_file')
    elif sum([generate, experiment, checkpoint, csv_file]) == 0:
        raise ValueError('One of the following options must be active: generate, experiment, checkpoint, csv_file')

    save = default_args['save'] if save is None else save
    file = default_args['file'] if file is None else file
    plot_losses = default_args['plot_losses'] if plot_losses is None else plot_losses

    # ----------------------------
    # configuration of plotter
    # ----------------------------

    n_conditions = default_args['n_conditions'] if n_conditions is None else n_conditions
    n_samples = default_args['n_samples'] if n_samples is None else n_samples
    batch_size = default_args['batch_size'] if batch_size is None else batch_size
    rows = default_args['starting_row'] if starting_row is None else starting_row
    gan_or_emb = 'gan'  # GAN samples: 'gan'; Embedding network samples: 'emb'; 'emb' was not tested yet!

    # ----------------------------
    # data processing configuration
    # ----------------------------

    bandpass = default_args['bandpass'] if bandpass is None else bandpass
    mvg_avg = default_args['mvg_avg'] if mvg_avg is None else mvg_avg
    moving_average_window = default_args['mvg_avg_window'] if mvg_avg_window is None else mvg_avg_window
    norm_data = not plot_losses            # normalize data to the range [0, 1]
    sequence_length = 24        # length of the sequence

    if bandpass and mvg_avg:
        warnings.warn('Both filters are active ("mvg_avg" and "bandpass"). Consider using only one.')

    # ----------------------------
    # run program
    # ----------------------------

    title = 'generated training samples'  # title of plot --> Is adjusted automatically according to configuration
    stacked = True
    data = None                 # specified automatically according to configuration

    # setup and configure according to generation or loading
    if experiment or csv_file or checkpoint:
        # load experimental data
        load_file = True
    else:
        # do not load any file but generate samples
        load_file = False

    if generate:
        # generate samples
        # load generator
        if file.split(os.path.sep)[0] == file:
            # use default path
            path = 'trained_models'
            if file.endswith('.pt'):
                file = os.path.join(path, file)
            elif file is None:
                file = os.path.join(path, 'checkpoint.pt')
            else:
                raise ValueError("File must be either None for loading a checkpoint or a dictionary ending with .pt")
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        opt = state_dict['configuration']
        generator = models.TtsGenerator(seq_length=opt['seq_len_generated'],
                                        latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt['seq_len_generated'],
                                        patch_size=opt['patch_size'])
        generator.load_state_dict(state_dict['generator'])

        if isinstance(generator, models.TtsGenerator):
            raise RuntimeError('visualize_main.py is not compatible with the "generate" keyword, right now. Please use the "checkpoint" keyword instead.')

        # generate samples
        generate_samples = GenerateSamples(generator, opt['seq_len_generated'], opt['latent_dim'])
        data = generate_samples.generate_samples(n_samples, conditions=False)

        # set plotting settings
        load_file, gan_or_emb, n_conditions, title = False, 'gan', 1, 'generated samples'

    if experiment:
        load_file, n_conditions, gan_or_emb, title = False, 3, 'gan', 'experimental data'
        if file.split(os.path.sep)[0] == file:
            # use default path
            path = 'data'
            file = os.path.join(path, file)
        if not file.endswith('.csv'):
            raise ValueError("Please specify a csv-file holding the experimental data.")
        dataloader = Dataloader(path=file, sequence_length=sequence_length, norm_data=True)
        data = dataloader.get_data()

    if checkpoint:
        # Load data from state_dict
        # if filename extension is .pt --> load state_dict and get samples from it
        if not file.endswith('.pt'):
            raise ValueError("Please specify a .pt-file holding a dictionary with the training data.")
        if file.split(os.path.sep)[0] == file:
            # use default path
            path = 'trained_models'
            file = os.path.join(path, file)
        if not plot_losses:
            data = np.array(torch.load(file, map_location=torch.device('cpu'))['generated_samples'])
        else:
            d_loss = torch.load(file, map_location=torch.device('cpu'))['discriminator_loss']
            g_loss = torch.load(file, map_location=torch.device('cpu'))['generator_loss']
            data = np.array([d_loss, g_loss])
            title = 'training losses'
            stacked = False
            norm_data = False
        load_file = False  # Otherwise most recent file from directory 'generated_samples' will be loaded

    if csv_file:
        # Load training data from csv file
        if file.split(os.path.sep)[0] == file:
            # use default path
            path = 'generated_samples'
            file = os.path.join(path, file)
        if not file.endswith('.csv'):
            raise ValueError("Please specify a .csv-file holding the training data.")
        data = pd.read_csv(file, delimiter=',', index_col=0)
        load_file = False

    # setup plotter
    plotter = PlotterGanTraining(load_file=load_file, filename=file,
                                 gan_or_emb=gan_or_emb, n_conditions=n_conditions,
                                 get_original_dataset=False)

    if data is not None:
        plotter.set_dataset(data)

    if mvg_avg:
        # filter data with moving average from GenerateSamples class
        plotter.set_dataset(GenerateSamples.moving_average(plotter.get_dataset(), w=moving_average_window))

    if bandpass:
        # filter data with bandpass filter from TtsGeneratorFiltered class
        plotter.set_dataset(models.TtsGeneratorFiltered.filter(plotter.get_dataset(), scale=True))

    if norm_data:
        # normalize each sample along time axis
        plotter.set_dataset(GenerateSamples.normalize_data(plotter.get_dataset(), axis=1))

    # plot data
    if not plot_losses:
        plotter.set_title(title + f'; {file if file is not None else ""}')
        # plotter.set_y_lim(max=int(1.5*batch_size), min=0)
        # plotter.set_y_lim(max=50, min=-10)
        plotter.plot(stacked=stacked, batch_size=batch_size, n_samples=n_samples, rows=rows, save=save)
    else:
        plt.plot(plotter.get_dataset()[0, :], label='discriminator loss')
        plt.plot(plotter.get_dataset()[1, :], label='generator loss')
        plt.title(plotter.title)
        plt.legend()
        if save:
            file = os.path.join('plots', 'losses.png')
            plt.savefig('losses.png')
        else:
            plt.show()
