---hide:    -toc---# Function Parameters
`heres a [description].(https://github.com/AutoResearch/EEG-GAN/blob/main/helpers/system_inputs.py) in a code style block`
This page contains the default parameters for the functions in the eeggan package. The parameters are organized by function and are listed in a table with the parameter name and a description of the parameter.

## Autoencoder Training

The autoencoder training function is used to train an autoencoder on EEG data. The autoencoder is trained to learn a compressed representation of the data. It can be incorporated into a GAN to improve the quality of the generated samples. The function has the following parameters:

|                 | Description                                                                                   | Default                      |
|:----------------|:----------------------------------------------------------------------------------------------|:-----------------------------|
| activation      | Activation function of the AE decoder; Options: [relu, leakyrelu, sigmoid, tanh, linear]      | sigmoid                      |
| batch_size      | Batch size                                                                                    | 128                          |
| channels_out    | Size of the encoded channels                                                                  | 10                           |
| checkpoint      | Path to a pre-trained AE                                                                      |                              |
| data            | Path to the dataset                                                                           | data\gansEEGTrainingData.csv |
| ddp             | Activate distributed training                                                                 | False                        |
| hidden_dim      | Hidden dimension of the transformer                                                           | 256                          |
| kw_channel      | Keyword to detect used channels                                                               |                              |
| kw_time         | Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time      | Time                         |
| learning_rate   | Learning rate of the AE                                                                       | 0.0001                       |
| load_checkpoint | Load a pre-trained AE                                                                         | False                        |
| n_epochs        | Number of epochs to train for                                                                 | 100                          |
| num_heads       | Number of heads of the transformer                                                            | 8                            |
| num_layers      | Number of layers of the transformer                                                           | 2                            |
| sample_interval | Interval of epochs between saving samples                                                     | 100                          |
| save_name       | Name to save model                                                                            |                              |
| seed            | Set seed for reproducibility                                                                  |                              |
| target          | Target dimension (channel, time, full) to encode; full is recommended for multi-channel data; | full                         |
| time_out        | Size of the encoded timeseries                                                                | 10                           |
| train_ratio     | Ratio of training data to total data                                                          | 0.8                          |

## GAN Training

The GAN training function is used to train a Generative Adversarial Network (GAN) on EEG data. The GAN is trained to generate realistic samples of EEG data. The function has the following parameters:

|                  | Description                                                                              | Default                      |
|:-----------------|:-----------------------------------------------------------------------------------------|:-----------------------------|
| autoencoder      | Path to an autoencoder                                                                   |                              |
| batch_size       | Batch size                                                                               | 128                          |
| checkpoint       | Path to a pre-trained GAN                                                                |                              |
| data             | Path to a dataset                                                                        | data\gansEEGTrainingData.csv |
| ddp              | Activate distributed training                                                            | False                        |
| discriminator_lr | Learning rate for the discriminator                                                      | 0.0001                       |
| generator_lr     | Learning rate for the generator                                                          | 0.0001                       |
| hidden_dim       | Hidden dimension of the GAN components                                                   | 16                           |
| kw_channel       | Keyword to detect used channels                                                          |                              |
| kw_conditions    | ** Conditions to be used                                                                 |                              |
| kw_time          | Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time | Time                         |
| n_epochs         | Number of epochs                                                                         | 100                          |
| num_layers       | Number of layers of the GAN components                                                   | 4                            |
| patch_size       | Patch size of the divided sequence                                                       | 20                           |
| sample_interval  | Interval of epochs between saving samples                                                | 100                          |
| save_name        | Name to save model                                                                       |                              |
| seed             | Set seed for reproducibility                                                             |                              |

## Visualization

The visualization function is used to visualize the results of training an autoencoder or GAN. The function has the following parameters:

|                 | Description                                                                              | Default                |
|:----------------|:-----------------------------------------------------------------------------------------|:-----------------------|
| average         | Average over all samples to get one averaged curve (per condition, if any is given)      | False                  |
| channel_index   | **Index of the channel to be plotted; If -1, all channels will be plotted;               | -1                     |
| channel_plots   | Plot each channel in a separate column                                                   | False                  |
| comp_data       | Path to a csv dataset for comparison; comparison only for t-SNE or PCA;                  | data\ganAverageERP.csv |
| data            | Use samples from csv-file                                                                |                        |
| fft             | Use a FFT-histogram to visualize the frequency distribution of the data                  | False                  |
| kw_channel      | Keyword to detect used channels                                                          |                        |
| kw_conditions   | ** Conditions to be used                                                                 |                        |
| kw_time         | Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time | Time                   |
| loss            | Plot training loss                                                                       | False                  |
| model           | Use samples from checkpoint file                                                         |                        |
| n_samples       | Total number of samples to be plotted                                                    | 0                      |
| pca             | Use PCA to reduce the dimensionality of the data                                         | False                  |
| spectogram      | Use spectogram to visualize the frequency distribution of the data                       | False                  |
| tsne            | Use t-SNE to reduce the dimensionality of the data                                       | False                  |
| tsne_iterations | Number of iterations of t-SNE                                                            | 1000                   |
| tsne_perplexity | Perplexity of t-SNE                                                                      | 40                     |

## Generate Samples

The generate samples function is used to generate samples of EEG data using a trained GAN. The function has the following parameters:

|                      | Description                                                                                  | Default                      |
|:---------------------|:---------------------------------------------------------------------------------------------|:-----------------------------|
| conditions           | ** Specific numeric conditions                                                               |                              |
| kw_time              | Keyword for the time step of the dataset; to determine the sequence length                   | Time                         |
| model                | File which contains the trained model and its configuration                                  | trained_models\checkpoint.pt |
| num_samples_parallel | number of samples generated in parallel                                                      | 50                           |
| num_samples_total    | total number of generated samples                                                            | 1000                         |
| save_name            | File where to store the generated samples; If None, then checkpoint name is used             |                              |
| seed                 | Set seed for reproducibility                                                                 |                              |
| sequence_length      | total sequence length of generated sample; if -1, then sequence length from training dataset | -1                           |

