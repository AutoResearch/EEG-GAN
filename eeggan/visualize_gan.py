import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from eeggan.helpers import system_inputs
from eeggan.nn_architecture import models
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers.visualize_pca import visualization_dim_reduction
from eeggan.helpers.visualize_spectogram import plot_spectogram, plot_fft_hist

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

        filenames = []
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
                    # plot the real and the reconstructed sample from an embedding network
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
                filenames.append(os.path.join(path, f'{file}_{i}.png'))
                plt.savefig(filenames[-1], dpi=600)
            else:
                plt.show()

        return filenames

    def set_dataset(self, gen_samples, conditions=False):
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

    def get_dataset(self, labels=False):
        dataset = self.df.to_numpy()
        # if not column:
        #     dataset = dataset[1:, :]
        # if not index:
        #     dataset = dataset[:, 1:]
        if not labels:
            dataset = dataset[:, self.n_conditions:]
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

def fun_plot_losses(d_loss, g_loss, save, title=None, path_save=None, legend=None):
    if legend is None:
        legend = ['Discriminator loss', 'Generator loss']
    plt.plot(d_loss, label=legend[0])
    plt.plot(g_loss, label=legend[1])
    plt.title(title)
    plt.legend()
    if save:
        if path_save is None:
            path_save = 'losses.png'
            path_save = os.path.join('plots', path_save)
        plt.savefig(path_save, dpi=600)
    else:
        plt.show()

    return d_loss, g_loss


def fun_plot_averaged(data, save=False, path_save=None):
    data = data.mean(axis=0)
    plt.plot(data)
    if save:
        if path_save is None:
            path_save = 'averaged.png'
            path_save = os.path.join('plots', path_save)
        plt.savefig(path_save, dpi=600)
    else:
        plt.show()

    return data

def visualize_gan(argv = []):

    #If run as a function, it receives a dictionary, which will here be converted to match terminal format
    #TODO: Do this in a more standard way?
    if isinstance(argv,dict):
        args = []
        for arg in argv.keys():
            if argv[arg] == True: #If it's a boolean with True
                args.append(str(arg)) #Only include key if it is boolean and true
            elif argv[arg] == False: #If it's a boolean with False
                pass #We do not include the argument if it is turned false
            else: #If it's not a boolean
                args.append(str(arg) + "=" + str(argv[arg])) #Include the key and the value
        argv = args
        
    # sys.argv = ["csv_file", "file=sd_len100_10000ep_cond0_20k.csv", "training_file=generated_samples\sd_len100_10000ep_cond1_20k.csv", "pca"]
    default_args = system_inputs.parse_arguments(argv, file='Visualize_Gan.py')

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    experiment = default_args['experiment']
    checkpoint = default_args['checkpoint']
    csv_file = default_args['csv_file']

    if sum([experiment, checkpoint, csv_file]) > 1:
        raise ValueError('Only one of the following options can be active: experiment, checkpoint, csv_file')
    elif sum([experiment, checkpoint, csv_file]) == 0:
        raise ValueError('One of the following options must be active: experiment, checkpoint, csv_file')

    save = default_args['save']
    # save_data = default_args['save_data']
    file = default_args['file']
    plot_losses = default_args['plot_losses']
    averaged = default_args['averaged']
    pca = default_args['pca']
    tsne = default_args['tsne']
    spectogram = default_args['spectogram']
    fft_hist = default_args['fft_hist']

    training_file = default_args['training_file']
    if training_file == training_file.split(os.path.sep)[0]:
        # get default file if only filename is given
        training_file = os.path.join('data', training_file)

    # ----------------------------
    # configuration of plotter
    # ----------------------------

    n_conditions = default_args['n_conditions']
    n_samples = default_args['n_samples']
    batch_size = default_args['batch_size']
    rows = default_args['starting_row']
    gan_or_emb = 'gan'  # GAN samples: 'gan'; Embedding network samples: 'emb'; 'emb' was not tested yet!
    legend = None

    # ----------------------------
    # data processing configuration
    # ----------------------------

    bandpass = default_args['bandpass']
    # mvg_avg = default_args['mvg_avg']
    # moving_average_window = default_args['mvg_avg_window']
    norm_data = not plot_losses            # normalize data to the range [0, 1]
    sequence_length = 24        # length of the sequence

    # if bandpass and mvg_avg:
    #     warnings.warn('Both filters are active ("mvg_avg" and "bandpass"). Consider using only one.')

    if pca and tsne:
        warnings.warn('Both dimensionality reduction methods are active ("pca" and "tsne"). "pca" will be used.')

    if plot_losses and (pca or tsne):
        raise RuntimeError('Dimensionality reduction methods cannot be used when plotting losses.')

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

    if experiment:
        load_file, n_conditions, gan_or_emb, title = False, len(default_args['conditions']), 'gan', 'experimental data'
        if file.split(os.path.sep)[0] == file:
            # use default path
            path = 'data'
            file = os.path.join(path, file)
        if not file.endswith('.csv'):
            raise ValueError("Please specify a csv-file holding the experimental data.")
        dataloader = Dataloader(path=file, kw_timestep=default_args['kw_timestep_dataset'], col_label=default_args['conditions'], norm_data=True)
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
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        if not plot_losses:
            data = np.array(state_dict['generated_samples'])
        else:
            state_dict = torch.load(file, map_location=torch.device('cpu'))
            keys = list(state_dict.keys())
            if 'discriminator_loss' in keys and 'generator_loss' in keys:
                d_loss = state_dict['discriminator_loss']
                g_loss = state_dict['generator_loss']
                legend = ['discriminator loss', 'generator loss']
            elif 'train_loss' in keys and 'test_loss' in keys:
                d_loss = state_dict['train_loss']
                g_loss = state_dict['test_loss']
                legend = ['train loss', 'test loss']
            elif 'loss' in keys:
                d_loss = np.array(state_dict['loss'])[:, 0].tolist()
                g_loss = np.array(state_dict['loss'])[:, 1].tolist()
                legend = ['train loss', 'test loss']
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

    # if mvg_avg:
    #     # filter data with moving average from GenerateSamples class
    #     plotter.set_dataset(GenerateSamples.moving_average(plotter.get_dataset(), w=moving_average_window))

    if bandpass:
        # filter data with bandpass filter from TtsGeneratorFiltered class
        plotter.set_dataset(models.TtsGeneratorFiltered.filter(plotter.get_dataset(), scale=True))

    # if norm_data:
    #     # normalize each sample along time axis
    #     plotter.set_dataset(GenerateSamples.normalize_data(plotter.get_dataset(), axis=1))

    # plot data
    legend_data = None
    if plot_losses:
        filename = file.split(os.path.sep)[-1].split('.')[0] + '_losses.png'
        filename = os.path.join('plots', filename)
        legend_data = legend
        curve_data = fun_plot_losses(plotter.get_dataset()[0, :], plotter.get_dataset()[1, :], save, plotter.title, filename, legend)
    elif averaged:
        filename = file.split(os.path.sep)[-1].split('.')[0] + '_averaged.png'
        filename = os.path.join('plots', filename)
        legend_data = ['averaged']
        curve_data = fun_plot_averaged(plotter.get_dataset(), save, filename)
    elif spectogram:
        filename = file.split(os.path.sep)[-1].split('.')[0] + '_spectogram.png'
        filename = os.path.join('plots', filename)
        curve_data = plot_spectogram(plotter.get_dataset(), save, filename)
    elif fft_hist:
        filename = file.split(os.path.sep)[-1].split('.')[0] + '_fft_hist.png'
        filename = os.path.join('plots', filename)
        curve_data = plot_fft_hist(plotter.get_dataset(), save, filename)
    elif pca or tsne:
        try:
            ori_data = Dataloader(path=training_file, norm_data=True).get_data().unsqueeze(-1).detach().cpu().numpy()[:, n_conditions:, :]
        except Exception as e:
            print("Training file was not of type experiment data. Trying to load generated samples.")
            ori_data = pd.read_csv(training_file, delimiter=',', index_col=0).to_numpy()
            ori_data = ori_data.reshape(-1, ori_data.shape[1], 1)
        gen_data = plotter.get_dataset()
        gen_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1], 1)[:, n_conditions:, :]
        if ori_data.shape[1] > gen_data.shape[1]:
            ori_data = ori_data[:, :gen_data.shape[1], :]
        elif ori_data.shape[1] < gen_data.shape[1]:
            gen_data = gen_data[:, :ori_data.shape[1], :]
        if pca:
            filename = file.split(os.path.sep)[-1].split('.')[0] + '_pca.png'
            filename = os.path.join('plots',  filename)
            legend = ['original data', 'generated data']
            curve_data = visualization_dim_reduction(ori_data, gen_data, 'pca', save, filename)
        elif tsne:
            filename = file.split(os.path.sep)[-1].split('.')[0] + '_tsne.png'
            filename = os.path.join('plots',  filename)
            legend = ['original data', 'generated data']
            curve_data = visualization_dim_reduction(ori_data, gen_data, 'tsne', save, filename, perplexity=default_args['tsne_perplexity'], iterations=default_args['tsne_iterations'])
    else:
        plotter.set_title(title + f'; {file if file is not None else ""}')
        filename = plotter.plot(stacked=stacked, batch_size=batch_size, n_samples=n_samples, rows=rows, save=save)

    if save:
        if not isinstance(filename, list):
            filename = [filename]
        for f in filename:
            print(f"Saved plot to {f}")
            
if __name__ == '__main__':
    visualize_gan(sys.argv)
    