import os.path

import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from utils.ae_dataloader import create_dataloader


def get_highest_freq(x, t, dx, threshold=0.005, plot=False, plot_index=500):
    print('lets find the highest frequency')

    # Perform DFT
    freq_domain = np.zeros_like(dx)
    for i in range(dx.shape[1]):
        freq_domain[:, i] = np.fft.fft(x[:, i], axis=0)

    # Get frequency resolution and number of samples
    freq_resolution = 1 / (t[1] - t[0])
    num_samples = len(dx)

    # Compute the frequency axis
    freq_axis = np.linspace(0, freq_resolution / 2, num_samples // 2 + 1)

    # normalize frequency domain so magnitudes of each freq_domain array sum up to 1
    freq_domain = freq_domain / np.abs(freq_domain).sum(axis=0)  # np.max(np.abs(freq_domain))
    freq_domain_new = freq_domain[:num_samples // 2 + 1]

    # get the highest index of magnitude over 0
    max_index = np.zeros(freq_domain_new.shape[1], dtype=int)
    for i in range(len(max_index)):
        freq_domain_array = np.abs(freq_domain_new[:, i])
        freq_domain_array[freq_domain_array < threshold] = 0
        max_index[i] = np.where(freq_domain_array > 0)[0][-1] if len(np.where(freq_domain_array > 0)[0]) > 0 else 0
    max_index = np.max(max_index)
    # get the highest frequency
    max_freq = freq_axis[max_index]

    if plot:
        # plot the frequency domain
        plt.plot(freq_axis, np.abs(freq_domain[:num_samples // 2 + 1, plot_index]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f"Highest frequency: {max_freq}")
        # plt.xlim([0, 200])
        plt.show()

    return max_freq


def get_butterworth(x, dx, t, max_freq, n_filters=3, save=False, save_path=None, plot=False, plot_index=500):
    print('lets do some filtering')

    # define the sampling frequency and the cutoff frequencies for each filter
    fs = 1 / (t[1] - t[0])  # sampling frequency
    freq_bounds = np.linspace(0, max_freq, n_filters)

    # define the low-pass filter
    b_low, a_low = signal.butter(4, freq_bounds[1] / (fs / 2), 'low')
    # define the high-pass filter
    b_high, a_high = signal.butter(4, freq_bounds[-1] / (fs / 2), 'high')
    # define the band-pass filter
    b_band, a_band = signal.butter(4, [freq_bounds[1] / (fs / 2), freq_bounds[-1] / (fs / 2)], 'band')

    dx_filt_low = np.zeros_like(dx)
    dx_filt_high = np.zeros_like(dx)
    dx_filt_band = np.zeros_like(dx)
    x_filt_low = np.zeros_like(dx)
    x_filt_high = np.zeros_like(dx)
    x_filt_band = np.zeros_like(dx)
    for i in range(x.shape[1]):
        # apply the filter to the gradient
        dx_filt_low[:, i] = signal.filtfilt(b_low, a_low, dx[:, i])
        dx_filt_high[:, i] = signal.filtfilt(b_high, a_high, dx[:, i])
        dx_filt_band[:, i] = signal.filtfilt(b_band, a_band, dx[:, i])

        # mask the gradient with 0 for each time step where the original signal is 0
        dx_filt_low[x[:, i] == 0, i] = 0
        dx_filt_high[x[:, i] == 0, i] = 0
        dx_filt_band[x[:, i] == 0, i] = 0

        # integrate the filtered gradient
        x_filt_low[:, i] = np.cumsum(dx_filt_low[:, i], axis=0)
        x_filt_high[:, i] = np.cumsum(dx_filt_high[:, i], axis=0)
        x_filt_band[:, i] = np.cumsum(dx_filt_band[:, i], axis=0)

    # define bandpass filter
    # x_filt_band_list, dx_filt_band_list = [], []
    # freq_bounds_bandpass = freq_bounds[1:]
    # for i, e in enumerate(freq_bounds_bandpass[:-1]):
    #     b_band, a_band = signal.butter(4, [freq_bounds_bandpass[i] / (fs / 2), freq_bounds_bandpass[i+1] / (fs / 2)], 'band')
    #     # apply the filter to the gradient
    #     dx_filt_band = signal.filtfilt(b_band, a_band, dx).reshape(-1, x.shape[1])
    #     # integrate the filtered gradient
    #     x_filt_band = np.cumsum(dx_filt_band, axis=0)
    #     # gather bandpass results
    #     x_filt_band_list.append(x_filt_band)
    #     dx_filt_band_list.append(dx_filt_band)

    # apply the filter to the gradient
    # dx_filt_high = signal.filtfilt(b_high, a_high, dx).reshape(-1, x.shape[1])
    # # integrate the filtered gradient
    # x_filt_high = np.cumsum(dx_filt_high, axis=0)

    if save:
        dataset_name = os.path.basename(save_path).split('.')[0]
        dataset_path = os.path.dirname(save_path)

        # save the filtered signal as new datasets
        df = pd.DataFrame(x_filt_low, columns=dataset.columns, index=dataset.index)
        df.to_csv(os.path.join(dataset_path, f"{dataset_name}_lowpass.csv"))
        df = pd.DataFrame(x_filt_high, columns=dataset.columns, index=dataset.index)
        df.to_csv(os.path.join(dataset_path, f"{dataset_name}_highpass.csv"))
        df = pd.DataFrame(x_filt_band, columns=dataset.columns, index=dataset.index)
        df.to_csv(os.path.join(dataset_path, f"{dataset_name}_bandpass.csv"))

        # save the filter coefficients and configuration in a dictionary with torch.save
        filter_dict = {'b_low': b_low, 'a_low': a_low,
                       'b_high': b_high, 'a_high': a_high,
                       'b_band': b_band, 'a_band': a_band,
                       'configuration': {'threshold': threshold,
                                         'n_filters': n_filters,
                                         'freq_bounds': freq_bounds}}
        torch.save(filter_dict, os.path.join('..', 'filter', f"{dataset_name}_filter_dict.pt"))

    if plot:
        # plot the original signal and the filtered signal
        plt.plot(t, x[:, plot_index] - x[0, plot_index], label='Original Signal')
        plt.plot(t, x_filt_high[:, plot_index] + x_filt_low[:, plot_index], label='high pass')
        # for i, x_bp in enumerate(x_filt_band):
        plt.plot(t, x_filt_band[:, plot_index] + x_filt_low[:, plot_index], label=f"bandpass")
        plt.plot(t, x_filt_low[:, plot_index], label='low pass')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f"cutoff frequencies: {freq_bounds}")
        plt.legend()
        plt.show()

        # plot sum of all filters vs original signal
        fig, ax = plt.subplots()
        ax.plot(t, x[:, plot_index] - x[0, plot_index], label='Original Signal')
        sum_bandpass = np.zeros(x.shape[0])
        # for i, x_bp in enumerate(x_filt_band_list):
        #     sum_bandpass += x_bp
        ax.plot(t, x_filt_high[:, plot_index] + x_filt_low[:, plot_index] + x_filt_band[:, plot_index],
                label='Summed filters')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        plt.show()

        # plot dx and dx_filt
        plt.plot(t, dx[:, plot_index], label='dx')
        plt.plot(t, dx_filt_high[:, plot_index], label='high pass')
        # for i, dx_bp in enumerate(dx_filt_band_list):
        plt.plot(t, dx_filt_band[:, plot_index], label=f"bandpass")
        plt.plot(t, dx_filt_low[:, plot_index], label='low pass')
        plt.legend()
        plt.show()

    return (x_filt_low, x_filt_high, x_filt_band), (dx_filt_low, dx_filt_high, dx_filt_band), {'lowpass': (b_low, a_low),
                                                                                               'highpass': (b_high, a_high),
                                                                                               'bandpass': (b_band, a_band)}


def moving_average(x, win_len, dtype=np.array, mode='same'):
    filtered = np.zeros((x.shape[0]-win_len+1, x.shape[1]))

    if len(x.shape) == 2:
        for i in range(x.shape[-1]):
            filtered[:, i] = np.convolve(x[:, i], np.ones(win_len) / win_len, 'valid')
    elif len(x.shape) == 3:
        for batch in range(x.shape[0]):
            for array in range(x.shape[-1]):
                filtered[batch, :, array] = np.convolve(x[batch, :, array], np.ones(win_len) / win_len, 'valid')
    elif len(x.shape) == 1:
        filtered = np.convolve(x, np.ones(win_len) / win_len, 'valid')
    else:
        raise ValueError("x must be a 1D (sequence), 2D (sequence, features) or 3D (batch, sequence, features) array")

    if mode == 'same':
        # adapting the window length of the to the remaining fragments at the beginning and end of the signal
        pad_before = np.zeros((win_len//2, x.shape[1]))
        pad_after = np.zeros((win_len//2, x.shape[1]))

        for i in range(win_len//2):
            pad_before[i] = np.mean(x[:i*2+1], axis=0)
            pad_after[i] = np.mean(x[-(i*2+1):], axis=0)
        # pad_before = np.flip(pad_before, axis=0)
        pad_after = np.flip(pad_after, axis=0)

        filtered = np.concatenate((pad_before, filtered, pad_after), axis=0)

        if filtered.shape[0] > x.shape[0]:
            filtered = filtered[1:]

    if dtype == np.array:
        return filtered
    elif dtype == torch.Tensor:
        return torch.from_numpy(filtered)


if __name__ == "__main__":
    seq_len = [50]
    n_filters = 3
    threshold = 0.005
    plot = True
    plot_index = 0#500
    save = False
    dataset_path = r'..\stock_data\stocks_sp500_2010_2020.csv'

    # load the signal from dataset
    # dataset = pd.read_csv(dataset_path, index_col=0)
    # x = dataset.to_numpy()
    for sl in seq_len:
        dataset, _, _ = create_dataloader(dataset_path, sl, 32, train_ratio=1, standardize=False, differentiate=False)
        dataset = dataset.dataset.data[0]
        x = dataset.squeeze(0).detach().numpy()

        t = np.linspace(0, 1, len(x))

        # get the gradient of the signal
        # dx = np.diff(x, axis=0)
        # dx = np.concatenate((dx, dx[-1].reshape(1, -1)), axis=0)
        # normalize dx to be between -1 and 1
        # dx = (dx / np.max(np.abs(dx)) + 1) / 2

        # get butterworth filter coefficients
        # max_freq = get_highest_freq(x, t, dx)
        # print(f"max freq: {max_freq}")
        # x_filt, dx_filt, cf_dict = get_butterworth(x, t, dx)

        # test moving average filters on the signal
        win_len = [50]
        # try:
        for wl in win_len:
            # dx_filt = np.zeros(x.shape)
            # for i in range(x.shape[1]):
            x_filt = moving_average(x, np.min((wl, len(x))))
            # x_filt = torch.from_numpy(np.cumsum(dx_filt, axis=0)).float()

            # fig, axs = plt.subplots(5, 2)
            # for i in range(5):
            #     axs[i, 0].plot(dx[:, i], label='original')
            #     axs[i, 0].plot(t, dx_filt[:, i], label='filtered')
            #     axs[i, 1].plot(t, np.cumsum(dx[:, i], axis=0), label='original')
            #     axs[i, 1].plot(t, np.cumsum(dx_filt[:, i], axis=0), label='filtered')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.title(f"seq length: {sl}; window length: {wl}")
            # plt.legend()
            # plt.show()

            if plot:
                # plot the original signal and the filtered signal
                plt.plot(x[:200, plot_index], label='original')
                plt.plot(x_filt[:200, plot_index], label='filtered')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title(f"seq length: {sl}; window length: {wl}")
                plt.legend()
                plt.show()

            if save:
                df = pd.read_csv(dataset_path, index_col=0)
                df_col = df.columns
                df_idx = df.index
                df = pd.DataFrame(x_filt, columns=df_col, index=df_idx)
                df.to_csv(f"../stock_data/stocks_sp500_2010_2020_mvgavg{wl}.csv")
        # except:
        #     print(f"seq length: {sl}; window length: {wl} failed")

