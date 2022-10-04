# GAN-in-Neuro

This repository contains a GAN-framework for the investigation of a GAN's capability to generate neuroscientific data

Start the training procedure by running main.py

Feel free to contribute!

# Main scripts

This repository has 3 scripts which are executable from the terminal.

1. gan_training_main.py: This script starts the training procedure

2. visualize_main.py: This script visualizes the results from training or the experimental data

3. checkpoint_to_csv.py: This script is used to convert the data from the checkpoint files to csv-files (so the data can be processed outside of the repo's environment)

# Instructions

1. Download/install the whole repo

2. Open the terminal and change to the repo's directory

3. Read the tutorials carefully. You get them by giving the terminal-command: python script.py help

# Further information

Use DDP-Training (Distributed Data Parallel Framework from PyTorch) if you want to apply the procedure to several GPUs.
Each GPU will process the complete dataset during one epoch. Therefore, divide the number of epochs you would usually take for one GPU by the number of available GPUs to calculate the DDP-Training number of epochs (n_epochs = n_epochs/n_GPUs)




