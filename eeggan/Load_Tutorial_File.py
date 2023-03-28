import numpy as np
import torch
import pkg_resources

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
        
    #Determine file stream
    stream = pkg_resources.resource_stream(__name__, filename)
    
    #Load file
    if filename.split('.')[-1] == 'csv':
        data = np.genfromtxt(stream, delimiter=',', names = True)
        headers = data.dtype.names
    elif filename.split('.')[-1] == 'pt':
        data = torch.load(stream, map_location=torch.device('cpu'))
        headers = []
    else:
        data = []
        headers = []
        print('This datatype is not supported by this function')    
    
    return headers, data
        
if __name__ == '__main__':
    load_tutorial_file()