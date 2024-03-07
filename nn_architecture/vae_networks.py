import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt

class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, encoded_dim=25, activation='tanh', device=None, **kwargs):
        super().__init__()

        #Variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
        self.num_electrodes = None
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Activation: Although we have options, tanh and linear are the only ones that works effectively
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activation function of type '{activation}' was not recognized.")

        #Encoder
        encoding_layers = [
            nn.Linear(input_dim, hidden_dim),
            self.activation
        ]
        self._encode = nn.Sequential(*encoding_layers)

        #Distributions
        self.mu_refactor = nn.Sequential(
            nn.Linear(hidden_dim, encoded_dim),
        )

        self.sigma_refactor = nn.Sequential(
            nn.Linear(hidden_dim, encoded_dim),
        )

        #Decoder
        decoding_layers = [
            nn.Linear(encoded_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        ]
        self._decode = nn.Sequential(*decoding_layers)

    def encode(self, x):
        
        x = torch.flatten(x, start_dim=1)
        x = self._encode(x)
        #x = self.activation(self.linear_enc_in(x))
        #x = self.encoder(x)
        #x = self.activation(self.linear_enc_out(x))
        mu = self.mu_refactor(x)
        sigma = self.sigma_refactor(x)

        return mu, sigma
    
    def sample(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        z = torch.randn(std.size(0),std.size(1))
        z = z*std + mu

        return z

    def decode(self, x):

        x = self._decode(x)
        #x = self.linear_dec_in(x)
        #x = self.decoder(x)
        #x = self.linear_dec_out(x)
        #x = torch.sigmoid(x)

        x = x.reshape((x.shape[0], self.input_dim, self.num_electrodes))

        return x

    def forward(self, x):
        if self.num_electrodes == None:
            self.num_electrodes = x.shape[-1]

        mu, sigma = self.encode(x)
        z_reparametrized = self.sample(mu, sigma)
        x_reconstructed = self.decode(z_reparametrized)

        return x_reconstructed, mu, sigma
  
    def generate_samples(self, loader, condition=0, num_samples=2500):

        self.num_electrodes = next(iter(loader)).shape[-1]
        
        generated_samples = np.empty((0,self.input_dim+1,1))
        for i, x in enumerate(loader):
            y = x[:,[0],0].to(self.device)
            x = x[:,1:,:].to(self.device)
            mu, sigma = self.encode(x)
            z = mu + sigma * torch.randn_like(sigma)
            sample_y = y.reshape(y.shape[0], y.shape[1], self.num_electrodes)
            sample_decoded = self.decode(z)
            gen_sample = torch.concat((sample_y, sample_decoded), dim=1)
            gen_sample = gen_sample[gen_sample[:,0,0]==condition,:,:]
            generated_samples = np.vstack((generated_samples, gen_sample.detach().numpy())) 

        return generated_samples[:num_samples,:]
    
    def plot_samples(self, loader, epoch):

        empirical_samples = np.empty((0,self.input_dim+1,1))
        for i, x in enumerate(loader):
           empirical_samples = np.vstack((empirical_samples, x.detach().numpy())) 

        syn0 = self.generate_samples(loader, condition=0, num_samples=2500)[:,1:,:]
        syn1 = self.generate_samples(loader, condition=1, num_samples=2500)[:,1:,:]

        emp0 = empirical_samples[empirical_samples[:,0,0]==0,1:,0]
        emp1 = empirical_samples[empirical_samples[:,0,0]==1,1:,0]

        fig, ax = plt.subplots(1,2)
        ax[1].plot(np.mean(syn0, axis=0), alpha=.5)
        ax[1].plot(np.mean(syn1,axis=0), alpha=.5)
        ax[1].set_title('VAE-Generated')
        ax[0].plot(np.mean(emp0,axis=0), alpha=.5)
        ax[0].plot(np.mean(emp1,axis=0), alpha=.5)
        ax[0].set_title('Empirical')

        plt.savefig(f'generated_images/generated_average_ep{epoch}.png')
        plt.close()

        for _ in range(200):
            c0_sample = syn0[np.random.randint(0,len(syn0)),:]
            c1_sample = syn1[np.random.randint(0,len(syn1)),:]
            plt.plot(c0_sample, alpha=.1, label='c0', color='C0')
            plt.plot(c1_sample, alpha=.1, label='c1', color='C1')
        plt.savefig(f'generated_images/generated_trials_ep{epoch}.png')
        plt.close()

    def plot_losses(self, recon_losses, kl_losses, losses):

        fig, ax = plt.subplots(3)
        ax[0].plot(recon_losses)
        ax[0].set_title('Reconstruction Losses')

        ax[1].plot(kl_losses)
        ax[1].set_title('KL Losses')

        ax[2].plot(losses)
        ax[2].set_title('Losses')

        plt.savefig(f'generated_images/vae_loss.png')
        plt.close()