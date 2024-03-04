#VAE Model:
import torch
import torch.nn.functional as F
from torch import nn

#VAE Training:
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

    def __init__(self, input_dim, hidden_dim=256, output_dim=128, z_dim=20, refactor_dim=2, num_layers=3, num_heads=4, dropout=0.1, activation='tanh', **kwargs):
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
        self._encode = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

        #self.linear_enc_in = nn.Linear(input_dim, hidden_dim)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        #self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.linear_enc_out = nn.Linear(hidden_dim, output_dim)

        #Distributions
        self.mu_refactor = nn.Sequential(
            nn.Linear(output_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, refactor_dim)
        )

        self.sigma_refactor = nn.Sequential(
            nn.Linear(output_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, refactor_dim)
        )

        #Decoder
        self._decode = nn.Sequential(
            nn.Linear(refactor_dim, z_dim),
            nn.Tanh(),
            nn.Linear(z_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
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

if __name__ == '__main__':
    
    #Set hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 100 #28x28 (image) -> 100 for EEG
    hidden_dim = 200
    z_dim = 20
    refactor_dim = 2
    
    batch_size = 128
    n_epochs = 200
    learning_rate = 1e-3
    kl_alpha = .00001

    #Load Data
    dataloader = Dataloader('data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_SS100_Run00.csv', col_label='Condition', channel_label='Electrode', norm_data=True)
    dataset = dataloader.get_data()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim, refactor_dim=refactor_dim).to(device)
    model.num_electrodes = dataset.shape[-1]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    #Training
    loop = tqdm(range(n_epochs))
    for epoch in loop:
        for i, x in enumerate(train_loader):
           
           y = x[:,0,:]
           x = x[:,1:,:].to(device)
           x_reconstruction, mu, sigma = model(x)

           #Loss
           reconstruction_loss = loss_fn(x_reconstruction, x)
           kl_div = torch.mean(0.5 * torch.sum(torch.exp(sigma) + mu.pow(2) - 1 - sigma, dim=-1))
           loss = reconstruction_loss + kl_div*kl_alpha
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
        loop.set_postfix(loss=loss.item())

        if  epoch % int(n_epochs*.1) == 0:
            model.inference(dataset=dataset, index = epoch)
