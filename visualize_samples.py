import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import TtsGenerator as Generator
from dataloader import Dataloader
from generate_samples import GenerateSamples


class PlotGeneratedSamples:
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

        self.file = None
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
        else:
            print("No file loaded. Please use set_dataset to set the dataset.")

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

    def plot(self, stacked=False, batch_size=None, rows=None, n_samples=None):
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
            self.df = pd.DataFrame(gen_samples)
        else:
            self.df = gen_samples

    def get_dataset(self, column=False, index=False, n_conditions=0):
        dataset = self.df.to_numpy()
        if not column:
            dataset = dataset[1:, :]
        if not index:
            dataset = dataset[:, 1:]
        if n_conditions > 0:
            dataset = dataset[:, n_conditions:]
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
    # ----------------------------
    # configuration of program
    # ----------------------------

    # if generate, experimental_data, load_file = False
    # the program will get the samples from the checkpoint.pt file in the directory trained_models
    generate = False            # generate samples with generator
    experimental_data = False   # use samples of experimental data

    load_file = True           # load samples from file; Only used if generate and experimental_data are False

    # specify if specific file from dir 'generated_samples' should be visualized else None
    filename = 'state_dict_tts_lp_seq_init.pt'

    # ----------------------------
    # configuration of plotter
    # ----------------------------

    n_conditions = 1            # number of columns with conditions BEFORE actual signal starts
    gan_or_emb = 'gan'          # GAN samples: 'gan'; Embedding network samples: 'emb'; 'emb' was not tested yet!
    n_samples = 10              # number of linearly drawn samples from given dataset
    batch_size = 10           # number of samples plotted in one figure
    rows = 0                    # number of rows to skip in dataset; useful to skip samples from early training

    # ----------------------------
    # data processing configuration
    # ----------------------------

    norm_data = False            # normalize data to the range [0, 1]
    mvg_avg = False             # moving average filter with window size moving_average_window
    moving_average_window = 1   # window size for moving average
    sequence_length = 24        # length of the sequence

    # ----------------------------
    # run program
    # ----------------------------

    title = 'generated training samples'  # title of plot --> Is adjusted automatically according to configuration
    data = None                 # specified automatically according to configuration

    # setup and configure according to generation or loading
    if experimental_data and generate:
        raise RuntimeError('You can not generate samples and use experimental data at the same time.')

    if experimental_data and not generate:
        # load experimental data
        load_file = True
    elif generate:
        # do not load any file but generate samples
        load_file = False

    if generate:
        # generate samples
        # load generator
        path = 'trained_models'
        if filename.endswith('.pt'):
            file = os.path.join(path, filename)
        else:
            file = os.path.join(path, 'checkpoint.pt')
        state_dict = torch.load(file, map_location=torch.device('cpu'))
        generator = Generator(seq_length=sequence_length,
                              latent_dim=17,
                              patch_size=12)
        generator.load_state_dict(state_dict['generator'])
        # generate samples
        generate_samples = GenerateSamples(generator)
        data = generate_samples.generate_samples()

        # set plotting settings
        load_file, gan_or_emb, n_conditions, title = False, 'gan', 1, 'generated samples'

    if experimental_data:
        load_file, n_conditions, gan_or_emb, title = False, 3, 'gan', 'experimental data'
        filename = r"C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganAverageERP.csv"
        dataloader = Dataloader(path=filename, sequence_length=sequence_length, diff_data=False, std_data=False, norm_data=True)
        data = dataloader.get_data()

    # Load data from state_dict
    if not generate and not experimental_data and load_file and data is None:
        # if filename extension is .pt --> load state_dict and get samples from it
        if filename.endswith('.pt'):
            path = 'trained_models'
            data = np.array(torch.load(os.path.join(path, filename), map_location=torch.device('cpu'))['generated_samples'])
            # filename = None  # set filename to None to avoid new loading of file within class
            load_file = False  # Otherwise most recent file from directory 'generated_samples' will be loaded

    # Load data from checkpoint.pt
    if not generate and not experimental_data and not load_file and data is None:
        # get data from checkpoint
        path = 'trained_models'
        filename = 'checkpoint.pt'
        data = np.array(torch.load(os.path.join(path, filename), map_location=torch.device('cpu'))['generated_samples'])

    # setup plotter
    plotter = PlotGeneratedSamples(load_file=load_file, filename=filename,
                                   gan_or_emb=gan_or_emb, n_conditions=n_conditions,
                                   get_original_dataset=False)

    if data is not None:
        plotter.set_dataset(data)

    # filter data with moving average from GenerateSamples class
    plotter.set_dataset(GenerateSamples.moving_average(plotter.get_dataset(), w=moving_average_window))
    # normalize each sample along time axis
    plotter.set_dataset(GenerateSamples.normalize_data(plotter.get_dataset(), axis=1))

    # plot data
    plotter.set_title(title + f'; {filename if filename is not None else ""}')
    # plotter.set_y_lim(max=int(1.5*batch_size), min=0)
    # plotter.set_y_lim(max=50, min=-10)
    plotter.plot(stacked=True, batch_size=batch_size, n_samples=n_samples, rows=rows)
