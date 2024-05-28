import warnings

import numpy as np
import pandas as pd
import torch
from typing import Union, List

from matplotlib import pyplot as plt


class Dataloader:
    """class of Dataloader, which is responisble for:
    - loading data from csv file
    - transform data (e.g. standardize, normalize, differentiate) and save the parameters for inverse transformation
    - convert to tensor"""

    def __init__(self, path=None,
                 diff_data=False, std_data=False, norm_data=False,
                 kw_time='Time', kw_conditions='', kw_channel=None):#, multichannel: Union[bool, List[str]]=False):
        """Load data from csv as pandas dataframe and convert to tensor.

        Args:
            path (str): Path to csv file.
            diff_data (bool): Differentiate data.
            std_data (bool): Standardize data.
        """

        if path is not None:
            # Load data from csv as pandas dataframe and convert to tensor
            df = pd.read_csv(path)

            # reshape and filter data based on channel specifications
            channels = [0]
            if kw_channel != '':
                channels = df[kw_channel].unique()
                assert len(df)%len(channels)==0, f"Number of rows ({len(df)}) must be a multiple of number of channels ({len(channels)}).\nThis could be caused by missing data for some channels."
            n_channels = len(channels)
            self.channels = channels

            # get first column index of a time step
            n_col_data = [index for index in range(len(df.columns)) if kw_time in df.columns[index]]

            if not isinstance(kw_conditions, list):
                kw_conditions = [kw_conditions]

            # Get labels and data
            dataset = torch.FloatTensor(df.to_numpy()[:, n_col_data])
            n_labels = len(kw_conditions) if kw_conditions[0] != '' else 0
            labels = torch.zeros((dataset.shape[0], n_labels))
            if n_labels:
                for i, l in enumerate(kw_conditions):
                    labels[:, i] = torch.FloatTensor(df[l])

            if diff_data:
                # Diff of data
                dataset = dataset[:, 1:] - dataset[:, :-1]

            self.dataset_min = torch.min(dataset)
            self.dataset_max = torch.max(dataset)
            if norm_data:
                # Normalize data
                dataset = (dataset - self.dataset_min) / (self.dataset_max - self.dataset_min)

            self.dataset_mean = dataset.mean(dim=0).unsqueeze(0)
            self.dataset_std = dataset.std(dim=0).unsqueeze(0)
            if std_data:
                # standardize data
                dataset = (dataset - self.dataset_mean) / self.dataset_std

            # reshape data to separate electrodes --> new shape: (trial, sequence, channel)
            if len(self.channels) > 1:
                sort_index = df.sort_values(kw_channel, kind="mergesort").index
                dataset = dataset[sort_index].contiguous().view(n_channels, dataset.shape[0]//n_channels, dataset.shape[1]).permute(1, 2, 0)
                labels = labels[sort_index].contiguous().view(n_channels, labels.shape[0]//n_channels, labels.shape[1]).permute(1, 2, 0)
            else:
                dataset = dataset.unsqueeze(-1)
                labels = labels.unsqueeze(-1)

            # concatenate labels to data
            dataset = torch.concat((labels, dataset), 1)

            self.dataset = dataset
            self.labels = labels

    def get_data(self, shuffle=True):
        """returns the data as a tensor"""
        if shuffle:
            return self.dataset[torch.randperm(self.dataset.shape[0])]
        else:
            return self.dataset

    def get_labels(self):
        return self.labels

    def dataset_split(self, dataset=None, train_size=0.8, shuffle=True):
        """Split dataset into train and test set. Returns the indices of the split."""

        # Split dataset into train and test set
        train_size = int(train_size * dataset.shape[0])

        if shuffle:
            dataset = dataset[torch.randperm(self.dataset.shape[0])]

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        return train_dataset, test_dataset

    def downsample(self, target_sequence_length):
        """Downsample data to target_sequence_length"""

        # Downsample data
        step_size = self.dataset.shape[1] // target_sequence_length
        self.dataset = torch.concat((self.labels, self.dataset[:, self.labels.shape[1]::step_size]), dim=1)

    def get_mean(self):
        return self.dataset_mean

    def get_std(self):
        return self.dataset_std

    def _windows_slices(self, sequence, window_size, stride=5):
        """Create a moving window of size window_size with stride stride.
        The last window is padded with 0 if it is smaller than window_size.

        Args:
            sequence (iterable): Input sequence.
            window_size (int): Size of the window.
            stride (int): Stride of the window.

        Returns:
            torch.Tensor: Tensor of windows.
        """
        warnings.warn("This function is deprecated and will be removed in future releases.", DeprecationWarning)

        # Create a moving window of size window_size with stride stride
        n_labels = self.labels.shape[1]
        sequence = sequence[:, n_labels:]
        windows = torch.zeros(((sequence.shape[1] - window_size) // stride, sequence.shape[0], window_size + n_labels))
        last_index = sequence.shape[1] - (window_size + stride) + 1
        for i in range(0, last_index, stride):
            # print(f"from {i} to {i+window_size}")
            windows[- ((sequence.shape[1] - window_size - i) // stride)] = torch.cat(
                (self.labels, sequence[:, i:i + window_size]), dim=-1)
        if sequence.shape[1] < last_index + stride + window_size:
            # if last window is smaller than window_size, pad with 0
            last_window = torch.zeros_like(windows[-1])
            last_window[:, :n_labels] = self.labels
            last_window[:, n_labels:n_labels + sequence.shape[1] - i - window_size] = sequence[:, i + window_size:]
            windows[-1] = last_window
        return windows.contiguous().view(-1, n_labels + window_size)

    def inverse_norm(self, data):
        """Inverse normalize data. Used also for generated samples by the generator."""

        # Check if data is tensor or numpy array
        if not isinstance(data, torch.Tensor):
            dataset_max = self.dataset_max.detach().cpu().numpy()
            dataset_min = self.dataset_min.detach().cpu().numpy()
        else:
            dataset_max = self.dataset_max
            dataset_min = self.dataset_min

        # Inverse normalize data
        data = data * (dataset_max - dataset_min) + dataset_min
        return data

    def inverse_std(self, data):
        """Inverse standardize data. Used also for generated samples by the generator."""

        # Check if data is tensor or numpy array
        if not isinstance(data, torch.Tensor):
            dataset_mean = self.dataset_mean.detach().cpu().numpy()
            dataset_std = self.dataset_std.detach().cpu().numpy()
        else:
            dataset_mean = self.dataset_mean
            dataset_std = self.dataset_std

        # Inverse standardize data
        data = data * dataset_std + dataset_mean
        return data

    @staticmethod
    def inverse_diff(data, dim):
        """Inverse differentiate data. Used also for generated samples by the generator."""

        # Inverse diff of data
        if isinstance(data, torch.Tensor):
            data = torch.cumsum(data, dim=dim)
        else:
            data = np.cumsum(data, axis=dim)
        return data

    def to_csv(self, path):
        """Save data to csv file"""
        if self.dataset is None:
            raise ValueError("Dataset is None. Please load data first.")
        pd.DataFrame(self.dataset.detach().cpu().numpy()).to_csv(path)
