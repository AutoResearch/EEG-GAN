
import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.dataloader import Dataloader
from nn_architecture.models import TransformerAutoencoder, TransformerFlattenAutoencoder, TransformerDoubleAutoencoder, train, save

datafile = 'data/ganTrialElectrodeERP_p50_e8_len100.csv'
modelfile = 'trained_ae/ae_ganTrialElectrodeERP_p50_e8_len100_1689868294.pth'

#Scale function
def scale(dataset):
    x_min, x_max = dataset.min(), dataset.max()
    return (dataset-x_min)/(x_max-x_min)

device = torch.device('cpu')

data = Dataloader(datafile, col_label='Condition', channel_label='Electrode')
batch = data.get_data()[0,1:,:].unsqueeze(0)
batch = scale(batch)
inputs = batch.float()

input_dim = inputs.shape[-1]
seq_length = inputs.shape[1]
    
model_dict = torch.load(modelfile, map_location=torch.device('cpu'))
model_state = model_dict['state_dict']

target = model_dict['config']['target']
channels_out = model_dict['config']['channels_out']
timeseries_out = model_dict['config']['timeseries_out']
    
if target == 'channels':
    model = TransformerAutoencoder(input_dim=input_dim, output_dim=channels_out).to(device)
elif target == 'timeseries':
    raise ValueError("Timeseries encoding target is not yet implemented")
elif target == 'full':
    #model = TransformerFlattenAutoencoder(input_dim=input_dim, sequence_length=seq_length, output_dim=channels_out).to(device) 
    model = TransformerDoubleAutoencoder(input_dim=input_dim, output_dim=channels_out, sequence_length=seq_length , output_dim_2=timeseries_out).to(device) 
else:
    raise ValueError(f"Encode target '{target}' not recognized, options are 'channels', 'timeseries', or 'full'.")
model.load_state_dict(model_state)

# decode a batch of sequences, rescale it with scaler and plot them
outputs = model.decode(model.encode(inputs.to(device)))

fig, axs = plt.subplots(10, 1, figsize=(10, 10), sharex=True)
batch_num = np.random.randint(0, batch.shape[0])
for i in range(10):
    feature = np.random.randint(0, inputs.shape[-1])
    axs[i].plot(inputs[batch_num, :, feature].detach().cpu().numpy(), label='Original')
    axs[i].plot(outputs[batch_num, :, feature].detach().cpu().numpy(), label='Reconstructed')
plt.legend()
plt.show()