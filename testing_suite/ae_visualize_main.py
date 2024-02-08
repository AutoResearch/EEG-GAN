
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader

#User input
data_checkpoint = 'data/ganTrialElectrodeERP_p500_e18_len100.csv'
ae_checkpoint = 'trained_ae/ae_ddp_4000ep_20230824_145643.pt'

#Load
ae_dict = torch.load(ae_checkpoint, map_location=torch.device('cpu'))
dataloader = Dataloader(data_checkpoint, col_label='Condition', channel_label='Electrode')
dataset = dataloader.get_data()
sequence_length = dataset.shape[1] - dataloader.labels.shape[1]
device = torch.device('cpu')

#Initiate
if ae_dict['configuration']['target'] == 'channels':
    ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_CHANNELS
    autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
elif ae_dict['configuration']['target'] == 'time':
    ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_TIMESERIES
    # switch values for output_dim and output_dim_2
    ae_output_dim = ae_dict['configuration']['output_dim']
    ae_dict['configuration']['output_dim'] = ae_dict['configuration']['output_dim_2']
    ae_dict['configuration']['output_dim_2'] = ae_output_dim
    autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
elif ae_dict['configuration']['target'] == 'full':
    autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], sequence_length=sequence_length, training_level=2).to(device)
    autoencoder.model_1 = TransformerDoubleAutoencoder(**ae_dict['configuration'], sequence_length=sequence_length, training_level=1).to(device)
    autoencoder.model_1.eval()
else:
    raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
autoencoder.load_state_dict(ae_dict['model'])
# freeze the autoencoder
for param in autoencoder.parameters():
    param.requires_grad = False
autoencoder.eval()

#Test
plt.figure()
plt.plot(ae_dict['train_loss'], label='Train Loss')
plt.plot(ae_dict['test_loss'], label = 'Test Loss')
plt.title('Losses')
plt.xlabel('Epoch')
plt.legend()
plt.show()

def norm(data):
    return (data-np.min(data)) / (np.max(data) - np.min(data))

dataset = norm(dataset.detach().numpy())

fig, axs = plt.subplots(5,1)
for i in range(5):
    sample = np.random.choice(len(dataset), 1)
    data = dataset[sample,1:,:]
    axs[i].plot(data[0,:,0], label='Original')
    axs[i].plot(autoencoder.decode(autoencoder.encode(torch.from_numpy(data)))[0,:,0].detach().numpy(), label='Reconstructed')
    axs[i].legend()
plt.show()

