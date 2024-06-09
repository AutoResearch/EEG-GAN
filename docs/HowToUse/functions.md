# GAN Package Main Functions

## <b>GAN Package Details</b>

EEG-GAN is a command line interface (CLI) package that allows users to train a Generative Adversarial Network (GAN) on EEG data. Once installed, you can run `eeggan` functions `<function>` in your terminal or command prompt alongside their parameters `<params>`: `eeggan <function> <params>`. <br><br>

`eeggan` has four functions:<br><br>
&emsp;&emsp;`gan_training` - This trains a GAN <br>
&emsp;&emsp;`autoencoder_training` - This trains an autoencoder <br>
&emsp;&emsp;`visualize` - This visualizes components of a trained GAN, such as the training losses <br>
&emsp;&emsp;`generate_samples` - This generates synthetic samples using the trained GAN<br>

<br><b>Arguments</b><br>

Each function can be followed by function parameters. Parameters are structured in that: <br>
&emsp;&emsp;Boolean arguments are passed as their argument name (e.g., `ddp`): <br>
&emsp;&emsp;&emsp;&emsp;`eeggan gan_training ddp` <br>
&emsp;&emsp;While other arguments are passed with an equals sign `=`: <br>
&emsp;&emsp;&emsp;&emsp;`eeggan gan_training data=data/eeg_training_data.csv`<br>
&emsp;&emsp; Arguments are separated by a space:<br>
&emsp;&emsp;&emsp;&emsp; `eeggan gan_training ddp data=data/eeg_training_data.csv`<br>

<br><b>Parameters</b><br>

You can use the help argument to see a list of possible parameters with a brief description:</b><br>
&emsp;&emsp;`eeggan gan_training help`<br>
&emsp;&emsp;`eeggan autoencoder_training help`<br>
&emsp;&emsp;`eeggan visualize help`<br>
&emsp;&emsp;`eeggan generate_samples help`<br>

You can also see these parameters the [Parameters](../parameters) page.