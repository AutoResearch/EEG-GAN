import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import nn

import system_inputs
from models import TtsDiscriminator, TtsGenerator
from get_master import find_free_port
from dataloader import Dataloader
from trainer_classifier import Trainer, DDPTrainer
from ddp_training_classifier import run

"""Train a classifier to distinguish samples between two conditions"""


if __name__ == "__main__":

    # sys.argv = ["generated", "path_dataset=generated_samples/sd_len100_10000ep.csv", "patch_size=20", "n_epochs=2", "sample_interval=1", "load_checkpoint", "path_test=trained_classifier\sd_onlyreal_10000ep.pt"]
    default_args = system_inputs.parse_arguments(sys.argv, system_inputs.default_inputs_training_classifier())

    if default_args['experiment'] and default_args['generated']:
        raise ValueError("The experiment and generated flags cannot be both True")

    if not default_args['experiment'] and not default_args['generated']:
        raise ValueError("The experiment and generated flags cannot be both False")

    if default_args['load_checkpoint']:
        print(f"Resuming training from checkpoint {default_args['path_checkpoint']}.")

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not default_args['ddp'] else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    opt = {
        'n_epochs': default_args['n_epochs'],
        'sequence_length': default_args['sequence_length'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'n_conditions': len(default_args['conditions']),
        'patch_size': default_args['patch_size'],
        'hidden_dim': 128,  # Dimension of hidden layers in discriminator and generator
        'world_size': world_size,  # number of processes for distributed training
        'device': device,
    }

    # if default_args['experiment']:
    # Load dataset as tensor
    if default_args['experiment']:
        # Get experiment's data as training data
        dataloader = Dataloader(default_args['path_dataset'],
                                kw_timestep=default_args['kw_timestep_dataset'],
                                col_label=default_args['conditions'],
                                norm_data=True)
        train_idx, test_idx = dataloader.dataset_split(train_size=.8)

        train_data = dataloader.get_data()[train_idx][:, dataloader.labels.shape[1]:]
        train_labels = dataloader.get_data()[train_idx][:, :dataloader.labels.shape[1]]
        test_data = dataloader.get_data()[test_idx][:, dataloader.labels.shape[1]:]
        test_labels = dataloader.get_data()[test_idx][:, :dataloader.labels.shape[1]]

    if default_args['generated']:
        # Get generated data as training data
        train_data = torch.tensor(pd.read_csv(default_args['path_dataset']).to_numpy()[:, opt['n_conditions']:]).float()
        train_labels = torch.tensor(pd.read_csv(default_args['path_dataset']).to_numpy()[:, :opt['n_conditions']]).float()
        if default_args['path_test'] != 'None':
            # Get test data if provided
            if default_args['path_test'].endswith('.pt'):
                # load checkpoint and extract test_dataset
                test_data = torch.load(default_args['path_test'], map_location=device)['test_dataset'][:, opt['n_conditions']:].float()
                test_labels = torch.load(default_args['path_test'], map_location=device)['test_dataset'][:, :opt['n_conditions']].float()
            elif default_args['path_test'].endswith('.csv'):
                # load csv
                test_data = torch.tensor(pd.read_csv(default_args['path_test']).to_numpy()[:, opt['n_conditions']:]).float()
                test_labels = torch.tensor(pd.read_csv(default_args['path_test']).to_numpy()[:, :opt['n_conditions']]).float()
        else:
            # Split train data into train and test
            dataloader = Dataloader()
            train_idx, test_idx = dataloader.dataset_split(train_data, train_size=.8)
            test_data = train_data[test_idx].view(train_data[test_idx].shape).float()
            test_labels = train_labels[test_idx].view(train_labels[test_idx].shape).float()
            train_data = train_data[train_idx].view(train_data[train_idx].shape).float()
            train_labels = train_labels[train_idx].view(train_labels[train_idx].shape).float()

    opt['sequence_length'] = train_data.shape[1]# - len(default_args['conditions'])

    if opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(
            f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
            f"The sequence is padded with zeros to fit the condition.")
        padding = 0
        while opt['sequence_length'] % default_args['patch_size'] != 0:
            padding += 1
        opt['sequence_length'] += padding
        train_dataset = torch.cat((train_data, torch.zeros(train_data.shape[0], padding)), dim=-1)
        test_dataset = torch.cat((test_data, torch.zeros(test_data.shape[0], padding)), dim=-1)

    # Load model and optimizer
    classifier = TtsDiscriminator(seq_length=opt['sequence_length'],
                                  patch_size=opt['patch_size'],
                                  n_classes=int(opt['n_conditions']*2),
                                  softmax=True).to(device)

    # Train model
    if default_args['ddp']:
        # DDP Training
        trainer = DDPTrainer(classifier, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        mp.spawn(run,
                 args=(world_size, find_free_port(), default_args['ddp_backend'], trainer, train_data, train_labels, test_data, test_labels),
                 nprocs=world_size, join=True)
    else:
        # Regular training
        trainer = Trainer(classifier, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        loss = trainer.train(train_data, train_labels, test_data, test_labels)

        # Save model
        path = 'trained_classifier'
        filename = 'classifier_' + timestamp + '.pt'
        filename = os.path.join(path, filename)
        trainer.save_checkpoint(filename, test_data, loss)

        print("Classifier training finished.")
        print("Model states, losses and test dataset saved to file: "
              f"\n{filename}.")
