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

    def inference(self, dataset, index, num_samples=5):

        idxs = torch.randint(0, len(dataset)-1, (num_samples, ))
        samples = torch.cat([dataset[idx,1:,:].reshape(1,self.input_dim,dataset.shape[-1]) for idx in idxs], dim=0).float()

        generated_samples, _, _ = model(samples)
        samples = (samples+1)/2
        generated_samples = 1 - (generated_samples + 1)/2

        fig, ax = plt.subplots(num_samples, 2)

        for sample_i in range(num_samples):
            ax[sample_i,0].plot(samples[sample_i,:,0].detach().numpy())
            ax[sample_i,1].plot(generated_samples[sample_i,:,0].detach().numpy())

        ax[0,0].set_title('Original')
        ax[0,1].set_title('Reconstructed')

        plt.savefig(f'generated_images/recon_ep{index}.png')

    def generate_samples(self, loader, epoch, num_samples=2500, plot=True):
        
        empirical_samples = np.empty((0,101,1))
        for i, x in enumerate(loader):
           empirical_samples = np.vstack((empirical_samples, x.detach().numpy())) 
        
        mus = np.empty((0,self.encoded_dim+1))
        sigmas = np.empty((0,self.encoded_dim+1))
        for i, x in enumerate(loader):
           y = x[:,[0],0].to(self.device)
           x = x[:,1:,:].to(self.device)
           mu, sigma = self.encode(x)
           mu_with_label = torch.concat((y,mu), dim=1)
           sigma_with_label = torch.concat((y,sigma), dim=1)

           mus = np.vstack((mus, mu_with_label.detach().numpy())) 
           sigmas = np.vstack((sigmas, sigma_with_label.detach().numpy())) 

        mu0 = torch.Tensor(np.mean(mus[mus[:,0]==0,1:], axis=0)) #TODO: Right now, it only looks at first electrode
        mu1 = torch.Tensor(np.mean(mus[mus[:,0]==1,1:], axis=0)) #TODO: Right now, it only looks at first electrode
        sigma0 = torch.Tensor(np.mean(sigmas[sigmas[:,0]==0,1:], axis=0)) #TODO: Right now, it only looks at first electrode
        sigma1 = torch.Tensor(np.mean(sigmas[sigmas[:,0]==1,1:], axis=0)) #TODO: Right now, it only looks at first electrode

        generated_samples = np.empty((0,101,1))
        for sample_index in range(num_samples):
            z0 = mu0 + sigma0 * torch.randn_like(sigma0)
            z1 = mu1 + sigma1 * torch.randn_like(sigma1)
            generated_sample0 = torch.concat((torch.Tensor([0]).reshape(1, -1, 1), self.decode(z0.reshape(1,-1))), dim=1) #TODO: Right now, it only assumes one electrode
            generated_sample1 = torch.concat((torch.Tensor([1]).reshape(1, -1, 1), self.decode(z1.reshape(1,-1))), dim=1) #TODO: Right now, it only assumes one electrode
            generated_sample = torch.concat((generated_sample0,generated_sample1), dim=0)
            generated_samples = np.vstack((generated_samples, generated_sample.detach().numpy())) 

        if not plot:
            return generated_samples
        
        syn0 = generated_samples[generated_samples[:,0,0]==0,1:,0] #TODO: Right now, it only looks at first electrode
        syn1 = generated_samples[generated_samples[:,0,0]==1,1:,0]
        emp0 = empirical_samples[empirical_samples[:,0,0]==0,1:,0]
        emp1 = empirical_samples[empirical_samples[:,0,0]==1,1:,0]

        fig, ax = plt.subplots(1,2)
        ax[1].plot(np.mean(syn0, axis=0), alpha=.5)
        ax[1].plot(np.mean(syn1,axis=0), alpha=.5)
        ax[0].plot(np.mean(emp0,axis=0), alpha=.5)
        ax[0].plot(np.mean(emp1,axis=0), alpha=.5)

        plt.savefig(f'generated_images/generated_average_ep{epoch}.png')
        plt.close()

        for _ in range(200):
            c0_sample = syn0[np.random.randint(0,len(syn0)),:]
            c1_sample = syn1[np.random.randint(0,len(syn1)),:]
            plt.plot(c0_sample, alpha=.1, label='c0', color='C0')
            plt.plot(c1_sample, alpha=.1, label='c1', color='C1')
        plt.savefig(f'generated_images/generated_trials_ep{epoch}.png')
        plt.close()

        return generated_samples

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