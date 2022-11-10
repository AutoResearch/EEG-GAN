import numpy as np
import torch
from matplotlib import pyplot as plt

from helpers.dataloader import Dataloader
from nn_architecture.models import TtsDiscriminator

if __name__ == "__main__":
    """Feed some generated samples into the discriminator and obtain the given scores. 
    The file needs to be a checkpoint file to load the GAN"""

    # load discriminator to check its performance
    file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\trained_models\sd_len100_train20_500ep.pt'
    dc = torch.load(file, map_location=torch.device('cpu'))
    critic = TtsDiscriminator(seq_length=dc['configuration']['sequence_length'],
                              patch_size=dc['configuration']['patch_size'],
                              in_channels=1 + dc['configuration']['n_conditions'])
    critic.load_state_dict(dc['discriminator'])
    critic.eval()

    # load data
    file = r'C:\Users\Daniel\PycharmProjects\GanInNeuro\data\ganTrialERP_len100_train.csv'
    data = Dataloader(file, norm_data=True).get_data()
    labels = data[:, :1]
    data = data[:, 1:]
    # get negated labels
    labels_neg = 1 - labels

    # get predictions
    for i in range(10):
        # random index
        idx = np.random.randint(0, data.shape[0], 10)
        # get data
        data_batch = data[idx, :]
        labels_batch = labels[idx, :]
        labels_neg_batch = labels_neg[idx, :]
        # get predictions
        labels = [labels_batch, labels_neg_batch]
        score = torch.zeros((labels_batch.shape[0], len(labels)))
        for j, label in enumerate(labels):
            batch_labels = label.view(-1, 1, 1, 1).repeat(1, 1, 1, data.shape[1])
            batch_data = data_batch.view(-1, 1, 1, data.shape[1])
            batch_data = torch.cat((batch_data, batch_labels), dim=1)
            validity = critic(batch_data)
            score[:, j] = validity[:, 0]
        plt.plot(score[:, 0].detach().numpy(), label='score real labels')
        plt.plot(score[:, 1].detach().numpy(), label='score negated labels')
        plt.plot(labels_batch.detach().numpy(), label='real labels')
        plt.legend()
        plt.show()