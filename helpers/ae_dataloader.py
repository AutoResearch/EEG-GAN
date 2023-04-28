import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, standardize=True, differentiate=False):
        self.seq_len = seq_len
        self.standardize = standardize
        self.differentiate = differentiate
        self.scaler = StandardScaler()
        self.data = self._process_data(data)

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.seq_len
        return self.data[start_index:end_index, :]

    def __len__(self):
        return self.data.shape[0] - self.seq_len + 1

    def _slice_sequence(self, sequence):
        slices = []
        num_slices = (len(sequence) - self.seq_len) + 1
        for i in range(num_slices):
            slice_start = i
            slice_end = i + self.seq_len
            slice_data = sequence[slice_start:slice_end]
            slices.append(slice_data)
        slices = torch.stack(slices[:-1], dim=0)
        return slices

    def _process_data(self, data):
        # convert the Date column to a datetime object and set it as the index
        # data['Date'] = pd.to_datetime(data['Date'])
        # data.set_index('Date', inplace=True)

        # differentiate the data if required
        if self.differentiate:
            data = data.diff().dropna()

        # standardize the data if required
        if self.standardize:
            data = pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns, index=data.index)

        # convert the data to a numpy array
        data = torch.Tensor(data.to_numpy())

        # slice the data into sequences
        self.data = self._slice_sequence(data)

        return data


def create_dataloader(training_data, seq_len, batch_size, train_ratio, standardize=True, differentiate=False, **kwargs):
    # load the data from the csv file
    data = pd.read_csv(training_data, index_col='Date')

    # split the data into train and test sets
    split_index = int(train_ratio * len(data))
    train_data = data.iloc[:split_index, :]
    test_data = data.iloc[split_index:, :]

    # create the datasets and dataloaders
    train_dataset = MultivariateTimeSeriesDataset(train_data, seq_len=seq_len, standardize=standardize, differentiate=differentiate)
    test_dataset = MultivariateTimeSeriesDataset(test_data, seq_len=seq_len, standardize=standardize, differentiate=differentiate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, train_dataset.scaler
