
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader

#Load function
def initiate_autoencoder(ae_checkpoint, dataset):
    ae_dict = torch.load(ae_checkpoint, map_location=torch.device('cpu'))

    n_channels = dataset.shape[-1]
    sequence_length = dataset.shape[1] - 1

    if ae_dict['configuration']['target'] == 'channels':
        autoencoder = TransformerAutoencoder(input_dim=n_channels,
                                       output_dim=ae_dict['configuration']['channels_out'],
                                       output_dim_2=sequence_length,
                                       target=TransformerAutoencoder.TARGET_CHANNELS,
                                       hidden_dim=ae_dict['configuration']['hidden_dim'],
                                       num_layers=ae_dict['configuration']['num_layers'],
                                       num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
    elif ae_dict['configuration']['target'] == 'time':
        autoencoder = TransformerAutoencoder(input_dim=sequence_length,
                                       output_dim=ae_dict['configuration']['timeseries_out'],
                                       output_dim_2=n_channels,
                                       target=TransformerAutoencoder.TARGET_TIMESERIES,
                                       hidden_dim=ae_dict['configuration']['hidden_dim'],
                                       num_layers=ae_dict['configuration']['num_layers'],
                                       num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
    elif ae_dict['configuration']['target'] == 'full':
        autoencoder = TransformerDoubleAutoencoder(input_dim=n_channels,
                                             output_dim=ae_dict['configuration']['output_dim'],
                                             output_dim_2=ae_dict['configuration']['output_dim_2'],
                                             sequence_length=sequence_length,
                                             hidden_dim=ae_dict['configuration']['hidden_dim'],
                                             num_layers=ae_dict['configuration']['num_layers'],
                                             num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
    else:
        raise ValueError(f"Encode target '{ae_dict['configuration']['target']}' not recognized, options are 'channels', 'time', or 'full'.")
    consume_prefix_in_state_dict_if_present(ae_dict['model'],'module.')
    autoencoder.load_state_dict(ae_dict['model'])
    autoencoder.device = torch.device('cpu')

    return ae_dict, autoencoder

#User input
data_checkpoint = 'data/ganTrialElectrodeERP_p100_e2_len100.csv'
ae_checkpoint = 'trained_ae/ae_ddp_1000ep_20240130_204203.pt'

#Load
dataloader = Dataloader(data_checkpoint, col_label='Condition', channel_label='Electrode')
dataset = dataloader.get_data()
sequence_length = dataset.shape[1] - dataloader.labels.shape[1]

#Initiate
ae_dict, autoencoder = initiate_autoencoder(ae_checkpoint, dataset)

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

