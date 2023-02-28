# GAN-in-Neuro

This repository contains a GAN-framework for the investigation of a GAN's capability to generate neuroscientific data

Start the training procedure by running main.py

Feel free to contribute!

# Main scripts

This repository has 3 scripts which are executable from the terminal.

1. gan_training_main.py: This script starts the training procedure

2. visualize_main.py: This script visualizes the results from training or the experimental data

3. generate_samples_main.py: This script initializes a trained generator to create synthetic samples

# Instructions

1. Download/install the whole repo

2. Open the terminal and change to the repo's directory

3. Read the tutorials carefully. You get them by giving the terminal-command: python script.py help

# Further information

Use DDP-Training (Distributed Data Parallel Framework from PyTorch) if you want to apply the procedure to several GPUs.
Each GPU will process the complete dataset during one epoch. Therefore, divide the number of epochs you would usually take for one GPU by the number of available GPUs to calculate the DDP-Training number of epochs (n_epochs = n_epochs/n_GPUs)

The training progress itself is saved in checkpoint.pt files and the final result has the name gan_xxxxx.pt
These files carry a python dictionary with all necessary information, models, generated samples, losses and so on.

# Running GANs on Brown's Oscar Cluster with 8GPUs

This method requires a different virtual environment than within the repo. Here are instructions on how to do this using Open on Demand (ood.ccv.brown.edu).

First, start a Virtual Desktop by going to the 'My Interactive Sessions' tab at the top and then selecting Desktop (Advanced). You will then be confronted with a range of fields with defaults. Change 'Partition' to 'GPU' and insert '8' under Num GPUs. You can also change the number of CPUs and RAM ize if you like, but defaults should work. Hit the 'Launch' button at the bottom when you are ready and it will bring you back to your 'My Interactive Sessions' tab with a session for 'Desktop (Advanced)' starting. The session will eventually establish (should not take long) and a 'Launch Desktop (Advanced)' button will appear.

Launching the desktop will take you to a virtual desktop. Open terminal and navigate to where you would like to create your virtual environment. You will then build the environment as such:

## Load modules
'''
module load python/3.9.0
module load gcc/10.2
module load cuda/11.7.1
module load cudnn/8.2.0
'''

## Create and activate virtual environment
'''
python3 -m venv myVirtualEnv

source ./myVirtualEnv/bin/activate
'''

## Install packages


