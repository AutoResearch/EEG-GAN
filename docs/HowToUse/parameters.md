# Function Parameters

This page contains the default parameters for the functions in the eeggan package. The parameters are organized by function and are listed in a table with the parameter name and a description of the parameter.

## Autoencoder Training

The autoencoder training function is used to train an autoencoder on EEG data. The autoencoder is trained to learn a compressed representation of the data. It can be incorporated into a GAN to improve the quality of the generated samples. The function has the following parameters:

|                 | Description                                                                                            | Default                          |
|:----------------|:-------------------------------------------------------------------------------------------------------|:---------------------------------|
| activation      | The activation function of the AE-Decoder. Options: [`relu`, `leakyrelu`, `sigmoid`, `tanh`, `linear`] | sigmoid                          |
| batch_size      | The batch size to use during training.                                                                 | 128                              |
| channels_out    | The number of channels in the encoded output.                                                          | 10                               |
| checkpoint      | The file which holds a pre-trained AE for further training.                                            |                                  |
| data            | The file which holds the EEG data to train the AE on.                                                  | data\eeggan_training_example.csv |
| ddp             | *Use the distributed data parallel training paradigm for training the AE on multiple GPUs.             | False                            |
| hidden_dim      | The dimension of the hidden layers in the AE.                                                          | 256                              |
| kw_channel      | The name of the channel column in the data to use for training the AE.                                 |                                  |
| kw_time         | The name of the time column in the data to use for training the AE.                                    | Time                             |
| learning_rate   | The learning rate for the AE.                                                                          | 0.0001                           |
| n_epochs        | The number of epochs to train the AE.                                                                  | 100                              |
| num_heads       | The number of attention heads to use in the transformer.                                               | 8                                |
| num_layers      | The number of hidden layers in the AE.                                                                 | 2                                |
| sample_interval | The epoch interval at which to save samples of the generated data during training.                     | 100                              |
| save_name       | The name to save the trained AE.                                                                       |                                  |
| seed            | The seed to use for reproducibility.                                                                   |                                  |
| target          | The name of the encoded dimension. Options: [`time`, `channel`, `full`]                                | full                             |
| time_out        | The number of time points in the encoded output.                                                       | 10                               |
| train_ratio     | The ratio of the data to use for training the AE vs. testing it.                                       | 0.8                              |

## GAN Training

The GAN training function is used to train a Generative Adversarial Network (GAN) on EEG data. The GAN is trained to generate realistic samples of EEG data. The function has the following parameters:

|                  | Description                                                                                                                                          | Default                          |
|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
| autoencoder      | The file which holds the trained autoencoder to use for training the AE-GAN.                                                                         |                                  |
| batch_size       | The batch size to use during training.                                                                                                               | 128                              |
| checkpoint       | The file which holds a pre-trained GAN for further training.                                                                                         |                                  |
| data             | The file which holds the EEG data to train the GAN on.                                                                                               | data\eeggan_training_example.csv |
| ddp              | *Use the distributed data parallel training paradigm for training the GAN on multiple GPUs.                                                          | False                            |
| discriminator_lr | The learning rate for the discriminator.                                                                                                             | 0.0001                           |
| generator_lr     | The learning rate for the generator.                                                                                                                 | 0.0001                           |
| hidden_dim       | The dimension of the hidden layers in the generator and discriminator.                                                                               | 16                               |
| kw_channel       | The name of the channel column in the data to use for training the GAN.                                                                              |                                  |
| kw_conditions    | **The names of the condition columns in the data to use for training the GAN.                                                                        |                                  |
| kw_time          | The name of the time column in the data to use for training the GAN.                                                                                 | Time                             |
| n_epochs         | The number of epochs to train the GAN.                                                                                                               | 100                              |
| num_layers       | The number of hidden layers in the generator and discriminator.                                                                                      | 4                                |
| patch_size       | The size of the patches used to split the data sequences (for more information read the TTS-GAN description (https://github.com/imics-lab/tts-gan)). | 20                               |
| sample_interval  | The epoch interval at which to save samples of the generated data during training.                                                                   | 100                              |
| save_name        | The name to save the trained GAN.                                                                                                                    |                                  |
| seed             | The seed to use for reproducibility.                                                                                                                 |                                  |

## VAE Training

The VAE training function is used to train a Variational Autoencoder (VAE) on EEG data. The VAE is trained to generate realistic samples of EEG data. The function has the following parameters:

|                 | Description                                                                                             | Default                                 |
|:----------------|:--------------------------------------------------------------------------------------------------------|:----------------------------------------|
| activation      | The activation function of the VAE-Decoder. Options: [`relu`, `leakyrelu`, `sigmoid`, `tanh`, `linear`] | tanh                                    |
| batch_size      | The batch size to use during training.                                                                  | 128                                     |
| checkpoint      | Path to a trained model to continue training                                                            |                                         |
| data            | The file which holds the EEG data to train the VAE on.                                                  | eeggan\data\eeggan_training_example.csv |
| encoded_dim     | The encoded dimension of mu and sigma used for generating new samples                                   | 25                                      |
| hidden_dim      | The dimension of the hidden layers in the VAE.                                                          | 256                                     |
| kl_alpha        | The weight of the KL divergence in the loss computation                                                 | 5e-05                                   |
| kw_channel      | The name of the channel column in the data to use for training the VAE.                                 |                                         |
| kw_conditions   | ** Conditions to be used                                                                                | Condition                               |
| kw_time         | The name of the time column in the data to use for training the VAE.                                    | Time                                    |
| learning_rate   | The learning rate for the VAE.                                                                          | 0.0003                                  |
| n_epochs        | The number of epochs to train the VAE.                                                                  | 1000                                    |
| sample_interval | The epoch interval at which to save samples of the generated data during training.                      | 100                                     |
| save_name       | The name to save the trained VAE.                                                                       |                                         |

## Visualization

The visualization function is used to visualize the results of training an autoencoder or GAN. The function has the following parameters:

|                 | Description                                                                                            | Default                |
|:----------------|:-------------------------------------------------------------------------------------------------------|:-----------------------|
| average         | *Whether to plot the average of the data.                                                              | False                  |
| channel_index   | **The index of the channels to visualize.                                                              | -1                     |
| channel_plots   | *Whether to plot the channels.                                                                         | False                  |
| comp_data       | The comparison data for `pca` or `tsne` to visualize.                                                  | data\ganAverageERP.csv |
| data            | The file containing the EEG data to visualize (only `model` or `data`).                                |                        |
| fft             | *Whether to plot FFT for visualization.                                                                | False                  |
| kw_channel      | The column name containing the channel label.                                                          |                        |
| kw_conditions   | **The column names with the conditions.                                                                |                        |
| kw_time         | The column names containing the time steps.                                                            | Time                   |
| loss            | *Whether to plot the loss of the model (`model` required).                                             | False                  |
| model           | The file containing the trained model with collected samples during training (only `model` or `data`). |                        |
| n_samples       | The number of samples to visualize.                                                                    | 0                      |
| pca             | *Whether to plot PCA for visualization (`comp_data` required).                                         | False                  |
| spectogram      | *Whether to plot a spectogram for visualization.                                                       | False                  |
| tsne            | *Whether to plot t-SNE for visualization (`comp_data` required).                                       | False                  |
| tsne_iterations | The number of iterations to use for t-SNE.                                                             | 1000                   |
| tsne_perplexity | The perplexity to use for t-SNE iterations.                                                            | 40                     |

## Generate Samples

The generate samples function is used to generate samples of EEG data using a trained GAN. The function has the following parameters:

|                      | Description                                                          | Default                      |
|:---------------------|:---------------------------------------------------------------------|:-----------------------------|
| conditions           | **The numeric condition values to use for generating samples.        |                              |
| kw_time              | The column name for the time steps to use for generating samples.    | Time                         |
| model                | The file containing the trained model to use for generating samples. | trained_models\checkpoint.pt |
| num_samples_parallel | The number of samples to generate in parallel (to speed things up).  | 50                           |
| num_samples_total    | The total number of samples to generate.                             | 1000                         |
| save_name            | The name for the csv file to save the generated samples.             |                              |
| seed                 | The seed to use for reproducibility.                                 |                              |
| sequence_length      | The length of the sequence to generate.                              | -1                           |

