
import numpy as np
import matplotlib.pyplot as plt
import torch
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder, TransformerFlattenAutoencoder
from helpers.dataloader import Dataloader

#User input
data_checkpoint = 'data/ganTrialElectrodeERP_p500_e18_len100.csv'
ae_checkpoint = 'trained_ae/ae_ddp_4000ep_20230824_145643.pt'

#Load
ae_dict = torch.load(ae_checkpoint, map_location=torch.device('cuda'))
dataloader = Dataloader(data_checkpoint, col_label='Condition', channel_label='Electrode')
dataset = dataloader.get_data()
sequence_length = dataset.shape[1] - dataloader.labels.shape[1]

#Initiate
if ae_dict['configuration']['model_class'] == 'TransformerAutoencoder':
    autoencoder = TransformerAutoencoder(**ae_dict['configuration'], sequence_length=sequence_length)
elif ae_dict['configuration']['model_class'] == 'TransformerDoubleAutoencoder':
    autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], sequence_length=sequence_length)
elif ae_dict['configuration']['model_class'] == 'TransformerFlattenAutoencoder':
    autoencoder = TransformerFlattenAutoencoder(**ae_dict['configuration'], sequence_length=sequence_length)
else:
    raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
autoencoder.load_state_dict(ae_dict['model'], strict=False)
autoencoder.device = torch.device('cpu')
print(ae_dict["configuration"]["history"])

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
    axs[i].plot(autoencoder.decode(autoencoder.encode(data))[0,:,0].detach().numpy(), label='Reconstructed')
    axs[i].legend()
plt.show()

