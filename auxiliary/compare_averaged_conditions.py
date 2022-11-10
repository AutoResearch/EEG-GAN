import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers.dataloader import Dataloader

if __name__ == "__main__":
    """Create a plot of two curves. Each one represents the averaged samples of one condition.
    The datasets to be processed have to be either in the directory: 
        - generated_samples if generated samples in the common csv-format (dimensions: (number samples, (condition, measurement)))
        - data if a study's dataset is taken"""

    # average over all samples

    # for generated sample
    file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\generated_samples\gan_train05_2500ep.csv'

    # for experiment files
    # file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganAverageERP_len100.csv'

    if os.path.sep + 'data' + os.path.sep in file:
        # file is in data folder is thus an experiment file
        data = Dataloader(file, norm_data=True).get_data(shuffle=False)
    elif os.path.sep + 'generated_samples' + os.path.sep in file:
        # file is in generated_samples folder is thus a generated sample file
        data = pd.read_csv(file)
        data = data.to_numpy()
    else:
        raise ValueError('File is not in data or generated_samples folder')

    # sort samples into respective bins
    data_cond0 = []
    data_cond1 = []
    for i in range(data.shape[0]):
        if data[i, 0] == 0:
            data_cond0.append(data[i, 1:].tolist())
        else:
            data_cond1.append(data[i, 1:].tolist())

    data_cond0 = np.array(data_cond0)
    data_cond1 = np.array(data_cond1)
    data_all = [data_cond1, data_cond0]

    erp = []
    legend = ['Condition 1', 'Condition 0']

    for i, f in enumerate(data_all):
        f = f.mean(axis=0)
        erp.append(f)
        plt.plot(f)
    plt.legend(legend)
    filename = os.path.basename(file).split('.')[0] + '_avg.png'
    filename = os.path.join(r'C:\Users\Daniel\PycharmProjects\GanInNeuro\plots', filename)
    # plt.savefig(filename)
    # plt.ylim((0.45, 0.6))
    plt.show()
