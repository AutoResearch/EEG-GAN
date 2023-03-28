import requests
import os

#Function to retrieve csv and pt files
def setup_tutorial(tutorial_file):
    
    #Hardcode file URLs to make it easier for users
    if tutorial_file == 'Training':
        url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/data/gansEEGTrainingData.csv'
        path = 'data/'
    elif tutorial_file == 'Validation':
        url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/data/gansEEGValidationData.csv'
        path = 'data/'
    elif tutorial_file == 'Samples':
        url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/generated_samples/gansEEGSyntheticData.csv'
        path = 'generated_samples/'
    elif tutorial_file == 'GAN':
        url = 'https://github.com/AutoResearch/EEG-GAN/raw/main/trained_models/gansEEGModel.pt'
        path = 'trained_models/'
    else:
        print('This file is not supported.')
        
    #Load the file
    r = requests.get(url, allow_redirects=True)

    #Create directory if needed
    if not os.path.exists(path):
        os.mkdir(path)
    
    #Save file to directory
    open(path+url.split('/')[-1], 'wb').write(r.content)

if __name__ == '__main__':
    setup_tutorial()