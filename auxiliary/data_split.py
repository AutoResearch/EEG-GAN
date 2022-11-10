import os

import numpy as np
import pandas as pd

from helpers.dataloader import Dataloader


if __name__ == "__main__":
    """Use this script to split the dataset into train and test data.
    Use the train data to train the GAN and the classifier.
    Use the test data to evaluate the classifier."""

    # setup
    file_dataset = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganTrialERP_len100.csv'
    conditions_dataset = ['Condition']  # the column name of the condition to train on
    n_data_col = 4  # number of column when the actual data begins
    train_size = 0.8
    shuffle_data = False

    # split dataset into train and test
    dataloader = Dataloader(file_dataset, col_label=conditions_dataset)
    train_data, test_data, train_idx, test_idx = dataloader.dataset_split(train_size=train_size, shuffle=shuffle_data)

    # train_data = dataloader.get_data()[train_idx][:, len(conditions_dataset):].detach().cpu().numpy()
    # test_data = dataloader.get_data()[test_idx][:, len(conditions_dataset):].detach().cpu().numpy()

    # load original data as dataframe
    df = pd.read_csv(file_dataset)
    columns = df.columns
    # get first n columns
    cond_train = df[columns[:n_data_col-1]].to_numpy()[train_idx]
    cond_test = df[columns[:n_data_col-1]].to_numpy()[test_idx]

    # to dataframe
    train_data = np.concatenate((cond_train, train_data[:, len(conditions_dataset):]), axis=1)
    test_data = np.concatenate((cond_test, test_data[:, len(conditions_dataset):]), axis=1)
    train_df = pd.DataFrame(train_data, columns=columns)
    test_df = pd.DataFrame(test_data, columns=columns)

    # save to csv
    path = os.path.dirname(file_dataset)
    file = os.path.basename(file_dataset)
    file = file.split('.')[0]
    filename_train = f'{file}_train.csv'
    filename_test = f'{file}_test.csv'
    train_df.to_csv(os.path.join(path, filename_train), index=False)
    test_df.to_csv(os.path.join(path, filename_test), index=False)

    print('Done')
