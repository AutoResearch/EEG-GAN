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
from nn_architecture.vae_networks import VariationalAutoencoder

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    #Set hyper-parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 100
    hidden_dim = 256
    encoded_dim = 25
    
    batch_size = 128
    n_epochs = 1000
    learning_rate = .0001
    kl_alpha = .00005
    activation = 'tanh'

    
    #Reset sample storage folder
    shutil.rmtree('generated_images')
    os.mkdir('generated_images')

    #Load Data

    dataloader = Dataloader('data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_SS100_Run00.csv', col_label='Condition', channel_label='Electrode')
    dataset = dataloader.get_data()
    norm = lambda data: (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    dataset[:,1:,:] = norm(dataset[:,1:,:]) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, encoded_dim=encoded_dim, activation=activation, device=device).to(device)
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
