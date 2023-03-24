# EEG-GAN

We here use Generative Adversarial Networks (GANs) to create trial-level synthetic EEG samples. We can then use these samples as extra data to train whichever classifier we want to use (e.g.,  Support Vector Machine, Neural Network).

You can find out documentation [here](https://autoresearch.github.io/EEG-GAN/)

Feel free to contribute!



# Running GANs on Brown's Oscar Cluster with 8GPUs (internal information for current developpers)

This method requires a different virtual environment than within the repo. Here are instructions on how to do this using Open on Demand (ood.ccv.brown.edu).

First, start a Virtual Desktop by going to the 'My Interactive Sessions' tab at the top and then selecting Desktop (Advanced). You will then be confronted with a range of fields with defaults. Change 'Partition' to 'GPU' and insert '8' under Num GPUs. You can also change the number of CPUs and RAM ize if you like, but defaults should work. Hit the 'Launch' button at the bottom when you are ready and it will bring you back to your 'My Interactive Sessions' tab with a session for 'Desktop (Advanced)' starting. The session will eventually establish (should not take long) and a 'Launch Desktop (Advanced)' button will appear.

Launching the desktop will take you to a virtual desktop. Open terminal and navigate to where you would like to create your virtual environment. You will then build the environment as such:

## Load modules
```
module load python/3.9.0
module load gcc/10.2
module load cuda/11.7.1
module load cudnn/8.2.0
```

## Create and activate virtual environment
```
python3 -m venv myVirtualEnv

source ./myVirtualEnv/bin/activate
```

## Install packages
Note that the following packages are all the same as the requirements.txt except for torch, torchvision, torchaudio, torchsummary. TODO: Add new requirements.txt
```
pip3 install torch torchvision torchaudio torchsummary 
pip install pandas==1.3.4
pip install numpy==1.21.4
pip install matplotlib==3.5.0
pip install scipy==1.8.0
pip install einops==0.4.1
pip install scikit-learn==1.1.2
```

## Run gans training
That should be all and now you should get no errors when running:
```
python gan_training_main.py ddp
```
