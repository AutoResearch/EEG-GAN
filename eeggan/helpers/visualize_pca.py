"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from helpers.dataloader import Dataloader


def visualization_dim_reduction(ori_data, generated_data, analysis, save, save_name=None, perplexity=40, iterations=1000, return_result=False):
    """Using PCA or tSNE for generated and original data visualization.
    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """

    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data), len(generated_data)])
    idx_ori = np.random.permutation(len(ori_data))[:anal_sample_no]
    idx_gen = np.random.permutation(len(generated_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx_ori]
    generated_data = generated_data[idx_gen]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    results_ori = None
    results_gen = None
    title = None
    xlabel = None
    ylabel = None

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        results_ori = pca.transform(prep_data)
        results_gen = pca.transform(prep_data_hat)

        title = 'PCA plot'
        xlabel = 'PC1'
        ylabel = 'PC2'

    elif analysis == 'tsne':
        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=iterations)
        tsne_results = tsne.fit_transform(prep_data_final)

        results_ori = tsne_results[:anal_sample_no, :]
        results_gen = tsne_results[anal_sample_no:, :]

        title = 't-SNE plot'
        xlabel = 'x-tsne'
        ylabel = 'y-tsne'

    if not return_result:
        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(results_ori[:, 0], results_ori[:, 1],
                    c='red', alpha=0.2, label="Original")
        plt.scatter(results_gen[:, 0], results_gen[:, 1],
                    c='blue', alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if save:
            if save_name is None:
                save_name = f'plots/{save_name}.png'
            plt.savefig(save_name, dpi=300)
        else:
            plt.show()

    return results_ori, results_gen


if __name__ == "__main__":

    ori_file = '../data/ganTrialERP_len100.csv'
    gen_file = '../generated_samples/sd_len100_19000ep.csv'
    return_result = True

    for arg in sys.argv:
        if not arg.endswith('.py'):
            if '=' in arg:
                arg = arg.split('=')[1]
                if arg[0] == 'original_data':
                    ori_file = arg[1]
                if arg[0] == 'generated_data':
                    gen_file = arg[1]

    # Load data
    dataloader = Dataloader(path=ori_file, norm_data=True)
    ori_data = dataloader.get_data().unsqueeze(-1).detach().cpu().numpy()
    # ori_data = np.load('data/real_data.npy')
    gen_data = pd.read_csv(gen_file, header=None).to_numpy()
    gen_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1], 1)

    if return_result:
        # split data into conditions
        ori_data_cond0 = ori_data[1:][ori_data[1:, 0, 0] == 0]
        ori_data_cond1 = ori_data[1:][ori_data[1:, 0, 0] == 1]
        gen_data_cond0 = gen_data[1:][gen_data[1:, 0, 0] == 0]
        gen_data_cond1 = gen_data[1:][gen_data[1:, 0, 0] == 1]

        ori_data = [ori_data_cond0, ori_data_cond1]
        gen_data = [gen_data_cond0, gen_data_cond1]
    else:
        ori_data = [ori_data]
        gen_data = [gen_data]

    results_real = []
    results_fake = []
    results_real_temp = None
    for i in range(len(ori_data)):
        # Visualization
        results_real_temp, results_fake_temp = visualization_dim_reduction(ori_data[i][:, 1:], gen_data[i][:, 1:], analysis='pca', save=True, return_result=return_result)
        results_real.append(results_real_temp)
        results_fake.append(results_fake_temp)

    if results_real_temp is not None:
        # Plotting
        f, ax = plt.subplots(1)
        for i in range(len(results_real)):
            plt.scatter(results_real[i][:, 0], results_real[i][:, 1],
                        c='red', alpha=0.2, label="Original")
            plt.scatter(results_fake[i][:, 0], results_fake[i][:, 1],
                        c='blue', alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
