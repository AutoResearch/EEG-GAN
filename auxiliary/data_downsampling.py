import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    """Sample a study's measurements down on a target length by linear interpolation.
    Save the new dataset in the data directory"""

    file = 'ganTrialERP.csv'
    target_sequence_length = 100  # target sequence length of the new dataset
    n_col_data = 2  # Where the actual measurement start

    path = '../data'

    # read data
    dataset = pd.read_csv(os.path.join(path, file))
    data = dataset.to_numpy()[:, n_col_data:].astype(np.float32)
    labels = dataset.to_numpy()[:, :n_col_data].astype(np.float32)

    # sampling on new grid
    x = np.linspace(0, data.shape[1], target_sequence_length)
    data_new = np.zeros((data.shape[0], target_sequence_length))
    for i in range(data.shape[0]):
        data_new[i] = np.interp(x, np.arange(data.shape[1]), data[i])
    data = data_new

    # make new dataframe
    dataset = pd.DataFrame(np.concatenate((labels, data), axis=1), columns=dataset.columns[:n_col_data + data.shape[1]])

    # save dataframe with column names and without index to csv file
    file = file.split('.')[0] + '_len' + str(target_sequence_length) + '.csv'
    dataset.to_csv(os.path.join(path, file), index=False)
