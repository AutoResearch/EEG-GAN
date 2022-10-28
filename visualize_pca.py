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

from dataloader import Dataloader


def visualization_dim_reduction(ori_data, generated_data, analysis, save, save_name=None, perplexity=40, iterations=1000):
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

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
    #         plt.show()

        # save results to csv files
        # if save:
        #     # if save_name is None:
        #     filename = 'pca_results.csv'
        #     columns = ['x_real', 'y_real', 'x_fake', 'y_fake']
        #     df = pd.DataFrame(np.concatenate((pca_results, pca_hat_results), axis=1), columns=columns)
        #     df.to_csv(filename, index=False)

    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=iterations)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
    #         plt.show()

    if save:
        if save_name is None:
            save_name = f'plots/{save_name}.png'
        plt.savefig(save_name, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":

    ori_file = 'data/ganTrialERP_len100.csv'
    gen_file = 'generated_samples/sd_len100_19000ep.csv'

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
    ori_data = dataloader.get_data().unsqueeze(-1).detach().cpu().numpy()[:, 1:, :]
    # ori_data = np.load('data/real_data.npy')
    gen_data = pd.read_csv(gen_file, header=None).to_numpy()[1:, 1:]
    gen_data = gen_data.reshape(gen_data.shape[0], gen_data.shape[1], 1)

    # Visualization
    visualization_dim_reduction(ori_data, gen_data, analysis='pca', save=True)
