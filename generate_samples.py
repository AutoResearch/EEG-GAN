import os

import numpy as np
import pandas as pd
import torch

from trainer import Trainer
from models import TtsGenerator, TtsGeneratorFiltered
from dataloader import Dataloader


class GenerateSamples:
    """This class is used to generate samples from a trained generator."""

    def __init__(self, generator, sequence_length=600, latent_dim=16):
        self.G = generator
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

    def sample_latent_variable(self, num_samples):
        seq_len = self.sequence_length if not isinstance(self.G, TtsGenerator) else 1
        return Trainer.sample_latent_variable(sequence_length=seq_len,
                                              batch_size=num_samples,
                                              latent_dim=self.latent_dim)

    def generate_samples(self, num_samples=10, conditions=False):
        z = self.sample_latent_variable(num_samples)
        labels = torch.randint(0, 2, (num_samples, 1))
        if not isinstance(self.G, TtsGenerator):
            y = self.G(z, labels)
        else:
            z = torch.cat((z, labels), dim=1)
            y = self.G(z)
        if conditions:
            y = torch.cat((labels, y.view(num_samples, self.sequence_length)), dim=1)
        return y

    @staticmethod
    def moving_average(x, w=10):
        result = np.zeros((x.shape[0], x.shape[1]-w+1))
        for i, v in enumerate(x):
            result[i] = np.convolve(v, np.ones(w)/w, 'valid')
        return result

    @staticmethod
    def normalize_data(x, axis=None):
        """Normalize the dataset."""
        # check if df has non-numeric values in the first row
        if isinstance(x[0, 0], str):
            x = x[1:, :].astype(float)
        if axis is None:
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        elif isinstance(axis, int):
            return (x - np.min(x, axis=axis).reshape(-1, 1)) / \
                   (np.max(x, axis=axis).reshape(-1, 1) - np.min(x, axis=axis).reshape(-1, 1))


if __name__ == '__main__':
    # file configuration
    filename_dataset = 'ganTrialERP.csv'  # dataset filename in directory 'data'
    filename_gen_samples = 'sample.csv'  # filename for generated samples, saved in directory 'generated_samples'
    filename_generator = None#'state_dict_tts_1400ep.pt'  # filename of the generator;

    num_samples_tot = 10  # total number of generated samples;
    # if num_samples_tot = None: num_samples_tot = number of samples in the dataset
    num_samples = 1        # number of samples being generated in parallel; set according to the processor capability
    # if num_samples_tot % num_samples != 0: num_samples_tot = num_samples * np.floor(num_samples_tot / num_samples)
    # If filename = none: generator is loaded from checkpoint.pth in the directory 'trained_models'

    # generator configuration
    generator_type = 'tts'  # type of generator: 'tts'; Other generators were not implemented in this script yet
    n_conditions = 1        # number of conditions
    latent_dim = 16         # latent dimension

    # load dataset
    print("Loading dataset...")
    path = 'data'
    dl = Dataloader(os.path.join(path, filename_dataset), False, False, True)
    dataset = dl.get_data()

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator
    print("Initializing generator...")
    generator = None
    if generator_type == 'tts':
        generator = TtsGeneratorFiltered(seq_length=dataset.shape[1]-n_conditions, latent_dim=latent_dim+n_conditions).to(device)

    # load generator weights
    path = 'trained_models'
    if filename_generator is None:
        filename_generator = 'checkpoint.pth'
    if generator is not None:
        print(f"Loading weights from {filename_generator}...")
        generator.load_state_dict(torch.load(os.path.join(path, filename_generator), map_location=device)['generator'])
    else:
        raise RuntimeError("Generator not defined.")

    # generate samples
    if num_samples_tot is None:
        num_sequences = int(np.floor(dataset.shape[0] / num_samples))
    else:
        num_sequences = int(np.floor(num_samples_tot / num_samples))
    latent_dim = generator.latent_dim - n_conditions
    gs = GenerateSamples(generator, sequence_length=dataset.shape[1]-n_conditions, latent_dim=latent_dim)
    all_samples = np.zeros((num_samples*num_sequences, dataset.shape[1]))
    print("Generating samples...")
    for i in range(num_sequences):
        print(f"Generating sequence {i+1} of {num_sequences}...")
        samples = gs.generate_samples(num_samples=num_samples, conditions=True).detach().cpu().numpy()
        all_samples[i*num_samples:(i+1)*num_samples, :] = samples

    # save samples
    print("Saving samples...")
    path = r'generated_samples'
    if filename_gen_samples is None:
        filename_gen_samples = 'generated_samples.csv'
    pd.DataFrame(all_samples).to_csv(os.path.join(path, filename_gen_samples), index=False, columns=None)
