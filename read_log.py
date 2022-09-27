import os

import numpy as np
import matplotlib.pyplot as plt


def get_loss_from_log(file_name, token_start='loss: ', token_end=']', token_training_line='[Epoch'):
    """read each line of log-file produced by GAN-training in main.py and output the D and G loss as floats.
    :param file_name: (String) name of log-file
    :param token_start: (String) token to start reading the loss from
    :param token_end: (String) token to end reading the loss from
    :param token_training_line: (String) token to identify the lines containing the loss"""

    g_loss_list = []
    d_loss_list = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(token_training_line):
                # get the first loss (should be discriminator loss)
                index_0 = line.find(token_start)
                index_1 = line.find(token_end, index_0)
                d_loss = float(line[index_0 + len(token_start):index_1])
                d_loss_list.append(d_loss)

                # get second loss (should be generator loss)
                index_0 = line.find(token_start, index_1)
                index_1 = line.find(token_end, index_0)
                g_loss = float(line[index_0 + len(token_start):index_1])
                g_loss_list.append(g_loss)
    return np.array(d_loss_list), np.array(g_loss_list)


if __name__ == '__main__':
    path = r'logs'
    filename = "log_tts_ws.log"
    d_loss, g_loss = get_loss_from_log(os.path.join(path, filename))
    d_loss_conv = np.convolve(d_loss, np.ones((1000,)) / 1000, mode='valid')
    # plt.plot(d_loss, label='discriminator loss')
    plt.plot(d_loss_conv, label='discriminator loss')

    g_loss_conv = np.convolve(g_loss, np.ones((1000,)) / 1000, mode='valid')

    plt.plot(g_loss_conv, label='generator loss')
    # plt.ylim(-np.max(g_loss), np.max(g_loss))
    plt.legend(['discriminator loss', 'generator loss'])
    plt.show()