import argparse
import sys

from .autoencoder_training_main import main as autoencoder_training_main
from .gan_training_main import main as gan_training_main
from .generate_samples_main import main as generate_samples_main
from .visualize_main import main as visualize_main
from .vae_training_main import main as vae_training_main

def main():

    #Extract command line arguments
    args = sys.argv
    if len(args) > 1:
        command = args[1]
        if len(args) > 2:
            args = args[2:]
        else:
            args = None
    else:
        ValueError('No command provided. Available commands are: gan_training, autoencoder_training, vae_training, generate_samples, visualize. Type "eeggan <command> help" for more information on a specific command.')
        sys.exit(1)

    #Parse command line arguments
    if not command:
        ValueError('No command provided. Available commands are: gan_training, autoencoder_training, vae_training, generate_samples, visualize. Type "eeggan <command> help" for more information on a specific command.')
        sys.exit(1)
    elif command.lower() == 'help':
        print('Available commands are: gan_training, autoencoder_training, vae_training, generate_samples, visualize. Type "eeggan <command> help" for more information on a specific command.')
    elif command == 'gan_training':
        gan_training_main(args)
    elif command.lower() == 'autoencoder_training':
        autoencoder_training_main(args)
    elif command.lower() == 'vae_training':
        vae_training_main(args)
    elif command.lower() == 'generate_samples':
        generate_samples_main(args)
    elif command.lower() == 'visualize':
        visualize_main(args)
    else:
        ValueError(f'Unrecognized command: {command}. Available commands are: gan_training, autoencoder_training, vae_training, generate_samples, visualize. ')
        sys.exit(1)
