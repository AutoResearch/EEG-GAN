import requests
import os

#Function to retrieve csv and pt files
def main():

    print('Downloading EEG-GAN tutorial files. Once completed, you will find the downloaded files in new directories that have been created during the process.')
    
    #Define file download function
    def download_file(path, url):
        #Load the file
        r = requests.get(url, allow_redirects=True)

        #Create directory if needed
        if not os.path.exists(path):
            os.mkdir(path)
        
        #Save file to directory
        open(path+url.split('/')[-1], 'wb').write(r.content)

        #Print success message
        print(f'{url.split("/")[-1]} has been downloaded and saved to directory {path.replace("/","")}.')

    #Create directories
    paths = ['data/', 'trained_ae/', 'trained_models/', 'generated_samples/', 'trained_vae/']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
    #Download EEG Training Data
    url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/dev/eeggan/data/eeggan_training_example.csv'
    path = 'data/'
    download_file(path, url)
    
    #Download EEG Validation Data
    url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/dev/eeggan/data/eeggan_validation_example.csv'
    path = 'data/'
    download_file(path, url)

    #Download EEG AE Model
    #url = 'https://github.com/AutoResearch/EEG-GAN/raw/main/trained_models/aeEEGModel.pt'
    #path = 'trained_ae/'
    #download_file(path, url)

    #Download EEG GAN Model
    #url = 'https://github.com/AutoResearch/EEG-GAN/raw/main/trained_models/gansEEGModel.pt'
    #path = 'trained_models/'
    #download_file(path, url)

    #Download EEG GAN Synthetic Data
    #url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/generated_samples/gansEEGSyntheticData.csv'
    #path = 'generated_samples/'
    #download_file(path, url)

    #Download EEG VAE Model
    #url = 'https://github.com/AutoResearch/EEG-GAN/raw/main/trained_models/vaeEEGModel.pt'
    #path = 'trained_vae/'
    #download_file(path, url)

    #Download EEG VAE Synthetic Data
    #url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/generated_samples/vaeEEGSyntheticData.csv'
    #path = 'generated_samples/'
    #download_file(path, url)

    print('EEG-GAN tutorial files have been downloaded.')

if __name__ == '__main__':
    main()