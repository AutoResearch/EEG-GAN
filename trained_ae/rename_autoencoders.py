import torch

import os
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

#RENAME EMBEDDED AUTOENCODER
ae_path = 'trained_ae/Reinforcement Learning/Embedded'
if not os.path.isdir(f"{ae_path}/renamed"):
    os.mkdir(f"{ae_path}/renamed") 

files = [f for f in listdir(ae_path) if isfile(join(ae_path, f))]

for file in tqdm(files):
    state_dict = torch.load(f"{ae_path}/{file}", map_location=torch.device('cpu'))
    new_name = state_dict['configuration']['path_dataset'].split('/')[-1].replace('ganTrialElectrodeERP','ae_ep2000').replace('csv','pt')
    torch.save(state_dict, f"{ae_path}/renamed/{new_name}")

#RENAME POST AUTOENCODER
ae_path = 'trained_ae/Reinforcement Learning/Post-GAN'
if not os.path.isdir(f"{ae_path}/renamed"):
    os.mkdir(f"{ae_path}/renamed") 

files = [f for f in listdir(ae_path) if isfile(join(ae_path, f))]

for file in tqdm(files):
    state_dict = torch.load(f"{ae_path}/{file}", map_location=torch.device('cpu'))
    new_name = state_dict['configuration']['path_dataset'].split('/')[-1].replace('augmentedData','post_ae_ep2000').replace('csv','pt')
    torch.save(state_dict, f"{ae_path}/renamed/{new_name}")

#RENAME GAN
ae_path = 'trained_models/Reinforcement Learning'
if not os.path.isdir(f"{ae_path}/renamed"):
    os.mkdir(f"{ae_path}/renamed") 

files = [f for f in listdir(ae_path) if isfile(join(ae_path, f))]

for file in tqdm(files):
    state_dict = torch.load(f"{ae_path}/{file}", map_location=torch.device('cpu'))
    new_name = state_dict['configuration']['path_dataset'].split('/')[-1].replace('ganTrialElectrodeERP','aegan_ep2000').replace('csv','pt')
    torch.save(state_dict, f"{ae_path}/renamed/{new_name}")

