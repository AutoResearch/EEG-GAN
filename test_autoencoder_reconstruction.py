
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nn_architecture.models import GANAE, train, save
from helpers.dataloader import Dataloader
import matplotlib.pyplot as plt

model = torch.load('trained_ae/ae_gansMultiCondition.pth')

#User inputs
filename = "data/gansMultiCondition.csv"
num_conditions = 1
num_epochs = 100

#Load and process data
data = Dataloader(filename, col_label='Condition', channel_label='Electrode')
dataset = data.get_data()[0,1:,:].unsqueeze(0).permute(0,2,1) #Get 

#Encode/decode data
encoded_data = model.encode(dataset).permute(0,2,1)
decoded_data = model.decode(encoded_data)

#Plot data
plt.plot(dataset[0,0,:].data, label='Original')
plt.plot(decoded_data[0,:,0].data, label='Reconstructed')
plt.legend()
plt.show()
