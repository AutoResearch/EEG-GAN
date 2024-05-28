#!/usr/bin/env python
import numpy as np
import pandas as pd
from helpers.dataloader import Dataloader
import torch
import warnings

warnings.filterwarnings("ignore")


def generate_fake_data(n_channels=1, label_channels=False, data_path=None):
    sequence_length = 600
    n_samples = 100
    data = np.zeros((n_samples * n_channels, sequence_length))
    df = pd.DataFrame(index=range(n_samples*n_channels), data=data,
                      columns=['Time'+str(i+1) for i in range(sequence_length)])
    for i in range(sequence_length):
        df['Time'+str(i+1)] = i+1
    df.insert(loc=0, column='ParticipantID', value=0)
    for idx in range(n_samples):
        df.iloc[idx*n_channels:(idx+1)*n_channels, 0] = idx
        # df.loc[(idx+1) * n_samples > df.index, ['ParticipantID']] = idx
    df.insert(loc=1, column='Electrode', value=0)
    for idx in range(n_channels):
        df.loc[df.index % n_channels == idx, ['Electrode']] = idx
    df.insert(loc=2, column='Condition', value=0)
    df.to_csv(data_path)
    return data_path


def run_test_reshaping_data(data_path, n_channels, ):
    dataloader = Dataloader(data_path,
                            kw_time='Time',
                            kw_conditions=['Condition', 'Electrode'],
                            n_channels=n_channels)
    dataset = dataloader.get_data(sequence_length=-1)
    # channel is in right dimension
    assert dataset.shape[2] == n_channels
    # for all samples in batch dimension
    for i in range(dataset.shape[0]):
        # confirm that the 1st column is still a list of the channels
        assert dataset[i, 1].mean() == torch.Tensor([n for n in range(n_channels)]).mean()
        assert dataset[i, 2:, 0].mean() == \
               torch.Tensor([n+1 for n in range(len(dataset[i, 2:, 0]))]).mean()


# generate fake data, where time series values are replaced with the value of the channel
# this test confirms that when data is reshaping, whole channels are preserved
def test_reshaping_data():
    # make fake data
    n_channels = 30
    data_path = generate_fake_data(n_channels=n_channels, data_path='../tests/data/test_multi-electrode.csv')
    run_test_reshaping_data(data_path=data_path, n_channels=n_channels)


if __name__ == '__main__':
    test_reshaping_data()
