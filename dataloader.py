import numpy as np
import pandas as pd
import torch


class Dataloader:
    """class of Dataloader, which is responisble for:
    - loading data from csv file
    - transform data (e.g. standardize, normalize, differentiate) and save the parameters for inverse transformation
    - convert to tensor"""

    def __init__(self, path, diff_data=False, std_data=False, norm_data=False, n_col_data=3, col_label='Condition'):
        """Load data from csv as pandas dataframe and convert to tensor.

        Args:
            path (str): Path to csv file.
            diff_data (bool): Differentiate data.
            std_data (bool): Standardize data.
        """

        # Load data from csv as pandas dataframe and convert to tensor
        dataset = pd.read_csv(path)
        labels = torch.FloatTensor(dataset[col_label]).unsqueeze(-1)
        dataset = torch.FloatTensor(dataset.to_numpy()[:, n_col_data:])

        if diff_data:
            # Diff of data
            dataset = dataset[:, 1:] - dataset[:, :-1]
        
        self.dataset_min = None
        self.dataset_max = None
        if norm_data:
            # Normalize data
            dataset_min = torch.min(dataset)
            dataset_max = torch.max(dataset)
            dataset = (dataset - dataset_min) / (dataset_max - dataset_min)
            self.dataset_min = dataset_min
            self.dataset_max = dataset_max
            
        self.dataset_mean = None
        self.dataset_std = None
        if std_data:
            # standardize data
            dataset_mean = dataset.mean(dim=0).unsqueeze(0)
            dataset_std = dataset.std(dim=0).unsqueeze(0)
            self.dataset_mean = dataset_mean
            self.dataset_std = dataset_std
            dataset = (dataset - dataset_mean) / dataset_std

        # concatenate labels to data
        dataset = torch.concat((labels, dataset), 1)

        self.dataset = dataset
        self.labels = labels

    def get_data(self):
        return self.dataset

    def get_labels(self):
        return self.labels

    def get_mean(self):
        return self.dataset_mean

    def get_std(self):
        return self.dataset_std

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
