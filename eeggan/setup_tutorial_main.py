import requests
import os

#Function to retrieve csv and pt files
def main():
    
    #Define file download function
    def download_file(path, url):
        #Load the file
        r = requests.get(url, allow_redirects=True)

        #Create directory if needed
        if not os.path.exists(path):
            os.mkdir(path)
        
        #Save file to directory
        open(path+url.split('/')[-1], 'wb').write(r.content)
    
    #Download each file
    url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/dev/eeggan/data/eeggan_training_example.csv'
    path = 'data/'
    download_file(path, url)
    
    url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/dev/eeggan/data/eeggan_validation_example.csv'
    path = 'data/'
    download_file(path, url)

    #url = 'https://raw.githubusercontent.com/AutoResearch/EEG-GAN/main/generated_samples/gansEEGSyntheticData.csv'
    #path = 'generated_samples/'
    #download_file(path, url)
    
    #url = 'https://github.com/AutoResearch/EEG-GAN/raw/main/trained_models/gansEEGModel.pt'
    #path = 'trained_models/'
    #download_file(path, url)

if __name__ == '__main__':
    main()