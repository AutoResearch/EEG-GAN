#Create a latex table with one column being name and the other column being description
import pandas as pd
from eeggan.helpers.system_inputs import default_inputs_training_autoencoder, default_inputs_training_gan, default_inputs_visualize, default_inputs_generate_samples
from collections import OrderedDict

#TODO: THE DESCRIPTIONS WERE AUTO-GENERATED, SO WE NEED TO WRITE THEM OURSELVES.

#Create a latex table from a dictionary
def create_latex_table(kw_dict):
    #Create a dictionary with the keys being the name of the variable and the values being the description
    kw_dict = OrderedDict(sorted(kw_dict.items(), key=lambda x: x[0]))
    kw_dict = {k: [v[1], v[2]] for k, v in kw_dict.items()}
    #Create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(kw_dict, orient='index', columns=['Description', 'Default'])

    return df

#Overwrite the descriptions of the default inputs for the GAN, autoencoder, visualization, and generation of samples functions
def overwrite_descriptions(function = 'GAN', kw_dict=None):

    '''
    Overwrite the descriptions of the default inputs for the GAN, autoencoder, visualization, and generation of samples functions.

    '''
    if function.lower() == 'gan':
        definitions = {
            'ddp': 'Use the distributed data parallel training paradigm for training the GAN on multiple GPUs.',
            'seed': 'The seed to use for reproducibility.',
            'n_epochs': 'The number of epochs to train the GAN.',
            'batch_size': 'The batch size to use during training.',
            'sample_interval': 'The epoch interval at which to save samples of the generated data during training.',
            'hidden_dim': 'The dimension of the hidden layers in the generator and discriminator.',
            'num_layers': 'The number of hidden layers in the generator and discriminator.',
            'patch_size': 'The size of the patches used to split the data sequences (for more information read [this description].(https://github.com/imics-lab/tts-gan)).',
            'discriminator_lr': 'The learning rate for the discriminator.',
            'generator_lr': 'The learning rate for the generator.',
            'data': 'The file which holds the EEG data to train the GAN on.',
            'checkpoint': 'The file which holds a pre-trained GAN for further training.',
            'autoencoder': 'The file which holds the trained autoencoder to use for training the AE-GAN.',
            'kw_conditions': '**The names of the condition columns in the data to use for training the GAN.',
            'kw_time': 'The name of the time column to use for training the GAN.',
            'kw_channel': 'The name of the channel column to use for training the GAN.',
            'save_name': 'The name to save the trained GAN.',
        }

    elif function.lower() == 'autoencoder':
        definitions = {
            'ddp': 'Whether to use DistributedDataParallel for training the autoencoder.',
            'load_checkpoint': 'Whether to load a checkpoint for the autoencoder.',
            'seed': 'The seed to use for reproducibility.',
            'data': 'The EEG data to train the autoencoder on.',
            'checkpoint': 'The directory to save the model checkpoints.',
            'save_name': 'The name to save the trained autoencoder.',
            'target': 'The target to use for training the autoencoder.',
            'kw_time': 'The time to use for training the autoencoder.',
            'kw_channel': 'The channel to use for training the autoencoder.',  
            'activation': 'The activation function to use in the autoencoder.',
            'channels_out': 'The number of channels in the output of the autoencoder.',
            'time_out': 'The number of time points in the output of the autoencoder.',
            'n_epochs': 'The number of epochs to train the autoencoder.',
            'batch_size': 'The batch size to use during training.',
            'sample_interval': 'The interval at which to save samples of the generated data.',
            'hidden_dim': 'The dimension of the hidden layers in the autoencoder.',
            'num_layers': 'The number of layers in the autoencoder.',
            'num_heads': 'The number of attention heads to use in the transformer.',
            'train_ratio': 'The ratio of the data to use for training the autoencoder.',
            'learning_rate': 'The learning rate for the autoencoder.',
        }

    elif function.lower() == 'visualization':
        definitions = {
            'loss': 'Whether to plot the loss of the model.',
            'average': 'Whether to plot the average of the data.',
            'pca': 'Whether to use PCA for visualization.',
            'tsne': 'Whether to use t-SNE for visualization.',
            'spectogram': 'Whether to use a spectogram for visualization.',
            'fft': 'Whether to use FFT for visualization.',
            'channel_plots': 'Whether to plot the channels.',
            'model': 'The trained model to visualize.',
            'data': 'The data to visualize.',
            'comp_data': 'The compressed data to visualize.',
            'kw_conditions': 'The conditions to use for visualization.',
            'kw_time': 'The time to use for visualization.',
            'kw_channel': 'The channel to use for visualization.',
            'n_samples': 'The number of samples to visualize.',
            'channel_index': 'The index of the channel to visualize.',
            'tsne_perplexity': 'The perplexity to use for t-SNE iterations.',
            'tsne_iterations': 'The number of iterations to use for t-SNE.',
        }

    elif function.lower() == 'generate_samples':
        definitions = {
            'seed': 'The seed to use for reproducibility.',
            'model': 'The trained model to use for generating samples.',
            'save_name': 'The name to save the generated samples.',
            'kw_time': 'The time to use for generating samples.',
            'sequence_length': 'The length of the sequence to generate.',
            'num_samples_total': 'The total number of samples to generate.',
            'num_samples_parallel': 'The number of samples to generate in parallel.',
            'conditions': 'The conditions to use for generating samples.',
        }

    else:
        print('Invalid function name. Please enter either "GAN", "autoencoder", "visualization", or "generate_samples".')

    #Overwrite the descriptions of the default inputs
    for key in kw_dict.keys():
        if key in definitions.keys():
            kw_dict[key][1] = definitions[key]

    return kw_dict

#Main function
def main(overwrite_desc=True):
    #Get the default inputs for the training of the autoencoder
    kw_dict = default_inputs_training_autoencoder()
    #Overwrite the descriptions of the default inputs for the autoencoder
    if overwrite_desc:
        kw_dict = overwrite_descriptions('autoencoder', kw_dict)
    #Create a latex table from the dictionaryw
    ae_table = create_latex_table(kw_dict)

    #Get the default inputs for the training of the GAN
    kw_dict = default_inputs_training_gan()
    #Overwrite the descriptions of the default inputs for the GAN
    if overwrite_desc:
        kw_dict = overwrite_descriptions('GAN', kw_dict)
    #Create a latex table from the dictionary
    gan_table = create_latex_table(kw_dict)

    #Get the default inputs for the visualization
    kw_dict = default_inputs_visualize()
    #Overwrite the descriptions of the default inputs for the visualization
    if overwrite_desc:
        kw_dict = overwrite_descriptions('visualization', kw_dict)
    #Create a latex table from the dictionary
    visualize_table = create_latex_table(kw_dict)

    #Get the default inputs for the generation of samples
    kw_dict = default_inputs_generate_samples()
    #Overwrite the descriptions of the default inputs for the generation of samples
    if overwrite_desc:
        kw_dict = overwrite_descriptions('generate_samples', kw_dict)
    #Create a latex table from the dictionary
    generate_table = create_latex_table(kw_dict)

    #Load and Save Tables as an .md file
    with open("docs/HowToUse/parameters.md", "w") as f:

        #Write the title of the markdown file
        f.write(f"# Function Parameters\n\n")
        f.write(f"This page contains the default parameters for the functions in the eeggan package. The parameters are organized by function and are listed in a table with the parameter name and a description of the parameter.\n\n")

        #Write the tables for the autoencoder, GAN, visualization, and generation of samples
        f.write(f"## Autoencoder Training\n\n")
        f.write(f"The autoencoder training function is used to train an autoencoder on EEG data. The autoencoder is trained to learn a compressed representation of the data. It can be incorporated into a GAN to improve the quality of the generated samples. The function has the following parameters:\n\n")
        f.write(f"{ae_table.to_markdown()}\n\n")

        f.write(f"## GAN Training\n\n")
        f.write(f"The GAN training function is used to train a Generative Adversarial Network (GAN) on EEG data. The GAN is trained to generate realistic samples of EEG data. The function has the following parameters:\n\n")
        f.write(f"{gan_table.to_markdown()}\n\n")

        f.write(f"## Visualization\n\n")
        f.write(f"The visualization function is used to visualize the results of training an autoencoder or GAN. The function has the following parameters:\n\n")
        f.write(f"{visualize_table.to_markdown()}\n\n")

        f.write(f"## Generate Samples\n\n")
        f.write(f"The generate samples function is used to generate samples of EEG data using a trained GAN. The function has the following parameters:\n\n")
        f.write(f"{generate_table.to_markdown()}\n\n")

if __name__ == '__main__':
    overwrite_desc = True #Whether to overwrite the descriptions of the default inputs
    main(overwrite_desc=overwrite_desc)

