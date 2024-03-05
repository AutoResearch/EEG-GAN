#VAE Model:
import torch
import torch.nn.functional as F
from torch import nn

#VAE Training:
import os
import shutil
import numpy as np
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from helpers.dataloader import Dataloader
from einops import rearrange
#from model import VariationalAutoencoder

import matplotlib.pyplot as plt

class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, z_dim=20, refactor_dim=2, num_layers=3, num_heads=4, dropout=0.1, activation='tanh', **kwargs):
        super().__init__()

        #Variables
        self.input_dim = input_dim

        #Other
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)

        self._encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            #nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers),
            #nn.Tanh(),
        )

        #self.linear_enc_in = nn.Linear(input_dim, hidden_dim)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        #self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.linear_enc_out = nn.Linear(hidden_dim, output_dim)

        #Distributions
        self.mu_refactor = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, refactor_dim)
        )

        self.sigma_refactor = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, refactor_dim)
        )

        self.decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        #Decoder
        self._decode = nn.Sequential(
            nn.Linear(refactor_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            #nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers),
            #nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        #self.linear_dec_in = nn.Linear(z_dim/2, hidden_dim)
        #self.decoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        #self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        #self.linear_dec_out = nn.Linear(hidden_dim, input_dim)

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
        z = torch.randn_like(std)
        z = z*std + mu

        return z

    def decode(self, x):

        x = self._decode(x)
        #x = self.linear_dec_in(x)
        #x = self.decoder(x)
        #x = self.linear_dec_out(x)
        #x = torch.sigmoid(x)

        x = x.reshape((x.shape[0], input_dim, self.num_electrodes))

        return x

    def forward(self, x):
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

    def generate_samples(self, loader, epoch):
        generated_samples = np.empty((0,101,1))
        for i, x in enumerate(loader):
                   
           y = x[:,[0],:].to(device)
           x = x[:,1:,:].to(device)
           mu, sigma = model.encode(x)
           z = mu + sigma * torch.randn_like(sigma)
           generated_sample = torch.concat((y,model.decode(z)), dim=1)
           generated_samples = np.vstack((generated_samples, generated_sample.detach().numpy()))  

        c0 = generated_samples[generated_samples[:,0,0]==0,1:,0] #TODO: Right now, it only looks at first electrode
        c1 = generated_samples[generated_samples[:,0,0]==1,1:,0] #TODO: Right now, it only looks at first electrode

        plt.plot(np.mean(c0,axis=0), alpha=.5)
        plt.plot(np.mean(c1,axis=0), alpha=.5)
        plt.savefig(f'generated_images/generated_average_ep{epoch-1}.png')
        plt.close()

        for _ in range(200):
            c0_sample = c0[np.random.randint(0,len(c0)+1),:]
            c1_sample = c1[np.random.randint(0,len(c1)+1),:]
            plt.plot(c0_sample, alpha=.1, label='c0', color='C0')
            plt.plot(c1_sample, alpha=.1, label='c1', color='C1')
        plt.savefig(f'generated_images/generated_trials_ep{epoch-1}.png')
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

if __name__ == '__main__':
    
    #Set hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 100 #28x28 (image) -> 100 for EEG
    hidden_dim = 200
    z_dim = 20
    refactor_dim = 10
    
    batch_size = 128
    n_epochs = 100
    learning_rate = 3e-3
    #kl_alpha = .00001
    kl_alpha = .00001

    #Reset sample storage folder
    shutil.rmtree('generated_images')
    os.mkdir('generated_images')

    #Load Data
    dataloader = Dataloader('data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e1_len100.csv', col_label='Condition', channel_label='Electrode')
    dataset = dataloader.get_data()
    norm = lambda data: (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    dataset[:,1:,:] = norm(dataset[:,1:,:]) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim, refactor_dim=refactor_dim).to(device)
    model.num_electrodes = dataset.shape[-1]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    #Training
    recon_losses=[]
    kl_losses=[]
    losses=[]

    loop = tqdm(range(n_epochs))
    for epoch in loop:
        for i, x in enumerate(train_loader):
           
            #Extract and reconstruct data
            y = x[:,0,:]
            x = x[:,1:,:].to(device)
            x_reconstruction, mu, sigma = model(x)

            #Loss
            reconstruction_loss = loss_fn(x_reconstruction, x)
            kl_div = torch.mean(-0.5 * torch.sum(1 + sigma - mu**2 - torch.exp(sigma), axis=1), dim=0)
            loss = reconstruction_loss + kl_div*kl_alpha
            
            #Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #Track losses
        recon_losses.append(reconstruction_loss.detach().tolist())
        kl_losses.append(kl_div.detach().tolist())
        losses.append(loss.detach().tolist())

        loop.set_postfix(loss=loss.item())

        if  epoch % int(n_epochs*.1) == 0:
        #    model.inference(dataset=dataset, index = epoch)
            model.generate_samples(loader=train_loader, epoch=epoch+1)
            model.plot_losses(recon_losses=recon_losses, kl_losses=kl_losses, losses=losses)

#model.inference(dataset=dataset, index = epoch+1)
        
model.generate_samples(loader=train_loader, epoch=epoch+2)
model.plot_losses(recon_losses=recon_losses, kl_losses=kl_losses, losses=losses)
