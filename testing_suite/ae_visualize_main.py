
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import tqdm
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, ReversedTransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader

#### User input ####
data_checkpoint = 'data/ganTrialElectrodeERP_p100_e2_len100.csv'
ae_checkpoint = 'trained_ae/ae_ddp_5000ep_p100_e8_enc50-4.pt'

#### Load data ####
dataloader = Dataloader(data_checkpoint, kw_conditions='Condition', kw_channel='Electrode')
dataset = dataloader.get_data().detach().numpy()
norm = lambda data: (data-np.min(data)) / (np.max(data) - np.min(data))
dataset = np.concatenate((dataset[:,[0],:], norm(dataset[:,1:,:])), axis=1)

#### Initiate autoencoder ####
device = torch.device('cpu')
ae_dict = torch.load(ae_checkpoint, map_location=device)
if ae_dict['configuration']['target'] == 'channels':
    ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_CHANNELS
    autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
elif ae_dict['configuration']['target'] == 'time':
    ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_TIMESERIES
    autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
elif ae_dict['configuration']['target'] == 'full':
    autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=2).to(device)
    autoencoder.model_1 = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=1).to(device)
    #autoencoder = ReversedTransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=2).to(device)
    #autoencoder.model_1 = ReversedTransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=1).to(device)
else:
    raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
autoencoder.load_state_dict(ae_dict['model'])
for param in autoencoder.parameters():
    param.requires_grad = False

#### Plot losses ####
plt.figure()
plt.plot(ae_dict['train_loss'], label='Train Loss')
plt.plot(ae_dict['test_loss'], label = 'Test Loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#### Plot trial level samples ####
fig, axs = plt.subplots(5,1)
for i in range(5):
    sample = np.random.choice(len(dataset), 1)
    data = dataset[sample,1:,:]
    axs[i].plot(data[0,:,0], label='Original')
    axs[i].plot(autoencoder.decode(autoencoder.encode(torch.from_numpy(data)))[0,:,0].detach().numpy(), label='Reconstructed')
    axs[i].legend()
plt.show()

#### Plot encoded-decoded averages ####

#Create reconstructed datasaet
ae_dataset = np.empty(dataset.shape)
print('Reconstructing dataset with the autoencoder...')
for sample in tqdm(range(dataset.shape[0])):
    data = dataset[[sample],1:,:]
    ae_data = autoencoder.decode(autoencoder.encode(torch.from_numpy(data))).detach().numpy()
    ae_dataset[sample,:,:] = np.concatenate((dataset[sample,0,:].reshape(1,1,-1), ae_data), axis=1)

fig, ax = plt.subplots(2,dataset.shape[-1])
for electrode in range(dataset.shape[-1]):
    ax[0,electrode].plot(np.mean(dataset[dataset[:,0,0]==0,1:,electrode],axis=0))
    ax[0,electrode].plot(np.mean(dataset[dataset[:,0,0]==1,1:,electrode],axis=0))

    ax[1,electrode].plot(np.mean(ae_dataset[ae_dataset[:,0,0]==0,1:,electrode],axis=0))
    ax[1,electrode].plot(np.mean(ae_dataset[ae_dataset[:,0,0]==1,1:,electrode],axis=0))
plt.show()




