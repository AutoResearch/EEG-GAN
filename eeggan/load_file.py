import numpy as np
import torch

#Function to retrieve csv and pt files
def load_file(filename):
    print(filename)
    print(filename.split('.')[-1])
    
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
    load_file()