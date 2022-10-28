import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dataloader import Dataloader


if __name__ == "__main__":

    # average over all samples

    # for generated samples
    data_all = [r'C:\Users\Daniel\PycharmProjects\GanInNeuro\generated_samples\sd_len100_30000ep_cond1.csv',
                r'C:\Users\Daniel\PycharmProjects\GanInNeuro\generated_samples\sd_len100_30000ep_cond0.csv']

    # for experiment files
    # data_all = [r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganTrialERP_len100.csv']
    # data = Dataloader(data_all[0], norm_data=True).get_data()
    # data_cond0 = []
    # data_cond1 = []
    # for i in range(data.shape[0]):
    #     if data[i, 0] == 0:
    #         data_cond0.append(data[i, 1:].tolist())
    #     else:
    #         data_cond1.append(data[i, 1:].tolist())
    #
    # data_cond0 = np.array(data_cond0)
    # data_cond1 = np.array(data_cond1)
    # data_all = [data_cond1, data_cond0]

    erp = []
    legend = ['Condition 1', 'Condition 0']

    for i, f in enumerate(data_all):
        # for generated samples
        f = pd.read_csv(f)
        f = f.to_numpy()[:, 1:]

        f = f.mean(axis=0)
        erp.append(f)
        plt.plot(f)
    plt.legend(legend)
    plt.savefig(r'C:\Users\Daniel\PycharmProjects\GanInNeuro\plots\ERP_averaged_generated_30000ep.png')
    plt.show()
