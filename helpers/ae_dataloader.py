import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, standardize=True, differentiate=False, shuffle=False):
        self.seq_len = seq_len
        self.standardize = standardize
        self.differentiate = differentiate
        self.scaler = StandardScaler()
        self.data = self._process_data(data, shuffle=shuffle)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

    def _slice_sequence(self, sequence):
        slices = []
        num_slices = (len(sequence) - self.seq_len) + 1
        for i in range(num_slices):
            slice_start = i
            slice_end = i + self.seq_len
            slice_data = sequence[slice_start:slice_end]
            slices.append(slice_data)
        if len(slices) > 1:
            slices = torch.stack(slices[:-1], dim=0)
        else:
            slices = slices[0].unsqueeze(0)
        return slices

    def _process_data(self, data, shuffle=False):
        # convert the Date column to a datetime object and set it as the index
        # data['Date'] = pd.to_datetime(data['Date'])
        # data.set_index('Date', inplace=True)

        # differentiate the data if required
        if self.differentiate:
            data = data.diff().dropna()

        # standardize the data if required but fit the scaler anyways
        data_std = self.scaler.fit_transform(data)
        if self.standardize:
            data = pd.DataFrame(data_std, columns=data.columns, index=data.index)

        # convert the data to a numpy array
        data = torch.Tensor(data.to_numpy())

        # slice the data into sequences
        if self.seq_len > 0:
            data = self._slice_sequence(data)
        else:
            data = data.unsqueeze(0)

        # shuffle the data along the batch dimension
        if shuffle:
            data = data[torch.randperm(data.shape[0])]

        return data

def create_dataloader(training_data, seq_len, batch_size, train_ratio, standardize=True, differentiate=False, shuffle=True, **kwargs):
    # load the data from the csv file
    data = pd.read_csv(training_data)
    
    # cut data to only include the time series
    time_index = [col for col in data if col.startswith('Time')]
    data = data[time_index]

    # split the data into train and test sets
    split_index = int(train_ratio * len(data))
    train_data = data.iloc[:split_index, :]
    test_data = data.iloc[split_index:, :]

    # if len of train or test data is less than seq_len throw error
    if train_ratio < 1.0 and (len(train_data) < seq_len or len(test_data) < seq_len):
        raise ValueError(f"Sequence length (={seq_len}) must be smaller than length of train data (={len(train_data)}) and length of test data (={len(test_data)}).")

    # create the datasets and dataloaders
    train_dataset, test_dataset, train_dataloader, test_dataloader = None, None, None, None
    if len(train_data) > 0:
        train_dataset = MultivariateTimeSeriesDataset(train_data, seq_len=seq_len, standardize=standardize, differentiate=differentiate, shuffle=shuffle)
    if len(test_data) > 0:
        test_dataset = MultivariateTimeSeriesDataset(test_data, seq_len=seq_len, standardize=standardize, differentiate=differentiate, shuffle=False)
    if train_dataset:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scaler = None
    if train_dataset:
        scaler = train_dataset.scaler
    elif test_dataset:
        scaler = test_dataset.scaler

    return train_dataloader, test_dataloader, scaler
