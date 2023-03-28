import numpy as np
import torch

#Function to retrieve csv and pt files
def load_tutorial_file(tutorial_file):
    
    #Hardcode files to make it easier for users (they shouldn't use this internal directory with their own data)
    if tutorial_file == 'Training':
        filename = 'data/gansEEGTrainingData.csv'
    elif tutorial_file == 'Validation':
        filename = 'data/gansEEGValidationData.csv'
    elif tutorial_file == 'Samples':
        filename = 'generated_samples/gansEEGSyntheticData.csv'
    elif tutorial_file == 'GAN':
        filename = 'trained_models/gansEEGModel.pt'
    else:
        print('This file is not supported.')
    
    #Load file
    if filename.split('.')[-1] == 'csv':
        headers = np.genfromtxt(filename, delimiter=',', names=True).dtype.names
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    elif filename.split('.')[-1] == 'pt':
        headers = []
        data = torch.load(filename, map_location=torch.device('cpu'))
    else:
        headers = []
        data = []
        print('This datatype is not supported by this function')    
    
    return headers, data
        
if __name__ == '__main__':
    load_tutorial_file()