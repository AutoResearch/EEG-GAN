import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import math

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#Load data
data = torchvision.datasets.StanfordCars(root=".", download=True)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding = 1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):

        #First conv
        h = self.bnorm(self.relu(self.conv1(x)))

        #Time embedding
        time_emb = self.relu(self.time_mlp(t))

        #Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]

        #Add time channel
        h = h + time_emb

        #Second conv
        h = self.bnorm(self.relu(self.conv2(h)))

        #Down or up sample
        return self.transform(h)

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)

        return embeddings

class diffusionModel(nn.Module):

    def __init__(self, device, T, batch_size):

        super().__init__()

        #Parameters
        self.device = device
        self.T = T
        self.image_size = 64
        self.batch_size = batch_size
        self.data_location = './'

        self.image_channels = 3
        self.down_channels = (64, 128, 256, 512, 1024)
        self.up_channels = (1024, 512, 256, 128, 64)
        self.out_dim = 1
        self.time_emb_dim = 32

        #Computations
        self.betas = self.linear_beta_schedule(timesteps=self.T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        #UNET Layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        #Initial projection
        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1)

        #Downsample
        self.downs = nn.ModuleList([Block(self.down_channels[i], self.down_channels[i+1], self.time_emb_dim) for i in range(len(self.down_channels)-1)])

        #Upsample
        self.ups = nn.ModuleList([Block(self.up_channels[i], self.up_channels[i+1], self.time_emb_dim, up=True) for i in range(len(self.up_channels)-1)])

        self.output = nn.Conv2d(self.up_channels[-1], 3, self.out_dim)

    def forward(self, x, timestep):

        #Embedd time
        t = self.time_mlp(timestep)

        #Initial conv
        x = self.conv0(x)

        #Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu()) #TODO: NEED CPU?
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device='cpu'):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def load_transformed_dataset(self):
        data_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #Scales to [0, 1]
            transforms.Lambda(lambda t: (t*2) -1) #Scales to [-1, 1]
        ]
        data_transform = transforms.Compose(data_transforms)

        train = torchvision.datasets.StanfordCars(root=self.data_location, download=True, transform=data_transform)
        test = torchvision.datasets.StanfordCars(root=self.data_location, download=True, transform=data_transform, split='test')

        return torch.utils.data.ConcatDataset([train,test])

    def show_tensor_image(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t+1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)), #CHW to HWC
            transforms.Lambda(lambda t: t * 255.),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ])

        if len(image.shape) == 4:
            image = image[0,:,:,:]

        plt.imshow(reverse_transforms(image))


#### TRAINING FUNCTIONS ####

def get_loss(model, x_0, t):
    x_noisy, noise = model.forward_diffusion_sample(x_0, t, model.device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = model.get_index_from_list(model.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = model.get_index_from_list(model.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = model.get_index_from_list(model.sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = model.get_index_from_list(model.posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image(model):
    img_size = model.image_size
    img = torch.randn((1, 3, img_size, img_size), device = model.device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(model.T/num_images)

    for i in range(0,model.T)[::-1]:
        t = torch.full((1,), i, device=model.device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize+1))
            model.show_tensor_image(img.detach().cpu())
    plt.show()

def main():
    batch_size = 256
    learning_rate = 0.001
    n_epochs = 10
    T = 200

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = diffusionModel(device, T, batch_size, )
    model.to(device)

    data = model.load_transformed_dataset()
    dataloader = DataLoader(data, model.batch_size, shuffle=True, drop_last=True)

    optimizer = Adam(model.parameters(), lr = learning_rate)

    loop = tqdm(range(n_epochs))
    for epoch in loop:
        for step, batch in tqdm(enumerate(dataloader)):

            t = torch.randint(0, model.T, (batch_size,), device=device).long()
            loss = get_loss(model, batch[0], t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                sample_plot_image(model)

            loop.set_postfix(step=f"{step+1}/{len(dataloader)}", loss=loss.item())

if __name__ == '__main__':
    main()