import torch
import pandas as pd
from nn_architecture.models import TtsGenerator
from helpers.dataloader import Dataloader
from helpers.trainer import Trainer
import numpy as np

# get images to be fixed
path_dataset = 'data/ganTrialElectrodeERP_mini_break10.csv'
data = pd.read_csv(path_dataset)

# initialized trained discriminator
file = 'trained_models/gan_ddp_5000ep_20221205_213325.pt'
state_dict = torch.load(file, map_location='cpu')
config = state_dict['configuration']
batch_size=config['batch_size']
latent_dim=config['latent_dim']
device=config['device']
sequence_length=config['sequence_length']
n_col_data = config['n_conditions']
n_conditions = n_col_data
channels=config['n_channels']
seq_len_cond=0
patch_size=config['patch_size']
generator = TtsGenerator(seq_length=sequence_length,
                         latent_dim=latent_dim + n_conditions + seq_len_cond,
                         patch_size=patch_size,
                         channels=channels).to(device)
generator.eval()
generator.load_state_dict(state_dict['generator'])

# transform dataset into suitable image format
del data['ParticipantID'], data['Trial'], data['Electrode']
#print(len(data.columns))

generated_samples = np.zeros(shape=(100*batch_size*channels,sequence_length))

df_numpy = data.to_numpy().reshape((data.shape[0]//channels, data.shape[1], channels))
dataset = torch.FloatTensor(df_numpy[:, n_col_data:])
labels = torch.zeros((data.shape[0], n_col_data))
for i, l in enumerate(['Condition']):
    labels[:, i] = torch.FloatTensor(data[l])
labels = labels.reshape((data.shape[0]//channels, n_col_data, channels))


for i in range(100):
    z = Trainer.sample_latent_variable(batch_size=batch_size, latent_dim=latent_dim,
                                       device=device, sequence_length=channels)
    z = z.reshape((batch_size, latent_dim, channels))
    gen_labels = torch.cat((labels, dataset), dim=1).to(device)
    z = torch.cat((z, gen_labels), dim=1)
    z = z.permute(0, 2, 1)
    gen_imgs = generator(z)[:batch_size]
    img = gen_imgs.numpy().reshape((batch_size,channels,sequence_length)).permute(0,2,1)
    data_batch = img.reshape((batch_size*channels,sequence_length))
    generated_samples[i*img.shape[0]:(i+1)*img.shape[0],:] = data_batch

generated_samples.tofile('generated_samples/channel10recovery.csv', sep=',')


# assert config['batch_size'] == 1
# data_numpy = data.to_numpy().reshape((config['batch_size'],
#                                       config['n_channels'],
#                                       1,
#                                       config['sequence_length'] + config['n_conditions']))
# img = data_numpy[:,:,:,config['n_conditions']:]
# img = torch.tensor(img)
# labels = torch.ones(img.shape)
# img = torch.cat((img, labels), dim=1).to(torch.device('cpu')).float()
#img = img.to(config['device'])
# print(img.shape)
# Improve image using discriminator criticism
# NOTE: Only channels indicated as dead_channels will change
# validity = discriminator(img)
# print(validity)
# print(validity.grad_fn)


# This code may help for batch size not equal to 1 - i.e. having to use a proper data loading process
# data = dataset[:, config['n_conditions']:].to(config['device'])
# data_labels = dataset[:, :config['n_conditions']].to(config['device'])
# gen_cond_data = data[:, :config['sequence_length']-config['sequence_length_generated']].to(config['device'])
# labels = data_labels.permute(0, 2, 1).view(-1, config['n_channels'], 1, 1).repeat(1, 1, 1, config['sequence_length']).to(config['device'])
# fake_data = torch.cat((img, labels), dim=1).to(config['device'])
# dataloader = Dataloader(config['path_dataset'],
#                             kw_timestep=config['kw_timestep_dataset'],
#                             col_label=config['conditions'],
#                             channels=config['n_channels'])
# dataset = dataloader.get_data(sequence_length=config['sequence_length'],
#                               pre_pad=config['sequence_length']-config['seq_len_generated'])