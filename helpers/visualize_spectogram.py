import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def plot_fft_hist(data, save=False, path_save=None):
    x = list(range(data.shape[1]))

    # apply fast fourier transform and take absolute values
    f = abs(np.fft.fft(data))

    # get the list of frequencies
    num = np.size(x)
    freq = [i / num for i in list(range(num))]

    # get the list of spectrums
    spectrum = f.real * f.real + f.imag * f.imag
    nspectrum = spectrum / spectrum[0]

    # make semilog 2D histogram from nspectrum
    ybins = 10 ** np.linspace(-3, 2, 100)
    xbins = np.array(x)
    y = nspectrum  # np.log10(nspectrum)
    x = np.array([x for i in range(data.shape[0])])

    h = np.zeros((len(xbins) - 1, len(ybins) - 1))
    for i in range(y.shape[0]):
        h += np.histogram2d(x[i], y[i], bins=(xbins, ybins))[0]

    # plot the histogram
    xbins = xbins[:-1] + (xbins[1] - xbins[0]) / 2
    ybins = ybins[:-1] + (ybins[1] - ybins[0]) / 2
    plt.pcolormesh(xbins, ybins, h.T, shading='gouraud')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.xlabel('Frequency [Hz]')
    plt.title('FFT Histogram')

    if save:
        if path_save is None:
            path_save = 'fft_hist.png'
        plt.savefig(path_save, dpi=600)
    else:
        plt.show()

    return xbins, ybins, h.T


def plot_spectogram(x, save=False, path_save=None):
    """Plot the spectogram of a dataset along the time axis (dim=1)."""

    fs = 500
    f, t, Sxx = signal.spectrogram(x.T, fs)
    Sxx = np.sum(Sxx, axis=0)

    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylim(10**-3, 50**1)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectogram')

    if save:
        if path_save is None:
            path_save = 'spectogram.png'
        plt.savefig(path_save, dpi=600)
    else:
        plt.show()

    return t, f, Sxx