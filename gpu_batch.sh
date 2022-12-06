#!/bin/bash
# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:2

#SBATCH --constraint=titanv|titanrtx|quadrortx

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH --mem 32G

#SBATCH -t 01:00:00

# Load Python
module load python/3.9.0
module load gcc/5.4
module load cuda/8.0.61

# Load Environment
source ./venv/bin/activate

# Run GANs Training
python gan_training_main.py ddp path_dataset=data/ganAverageERP_len100.csv n_epochs=500

