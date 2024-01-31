import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import torch
import torch.multiprocessing as mp

from helpers import system_inputs
from nn_architecture.models import TtsClassifier, TtsDiscriminator
from helpers.get_master import find_free_port
from helpers.dataloader import Dataloader
from helpers.trainer_3c import Trainer, DDPTrainer
from helpers.ddp_training_classifier import run

"""Train a classifier to distinguish samples between two conditions"""


if __name__ == "__main__":

    # sys.argv = ["experiment", "path_dataset=data\ganTrialERP_len100_test_shuffled.csv", "path_test=data\ganTrialERP_len100_train_shuffled.csv", "n_epochs=1", "sample_interval=10", "path_critic=trained_models\sd_len100_train20_500ep.pt"]#, "load_checkpoint", "path_checkpoint=trained_3c\\3c_exp_train20_3ep.pt"]
    # sys.argv = ["generated", "path_test=trained_classifier\\cl_exp_109ep.pt", "path_dataset=generated_samples\\sd_len100_10000ep.csv", "n_epochs=2", "sample_interval=10"]
    default_args = system_inputs.parse_arguments(sys.argv, system_inputs.default_inputs_training_classifier())

    if not default_args['experiment'] and not default_args['generated'] and not default_args['testing']:
        raise ValueError("At least one of the following flags must be set: 'experiment', 'generated', 'testing'.")

    if default_args['load_checkpoint']:
        print(f"Resuming training from checkpoint {default_args['path_checkpoint']}.")

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not default_args['ddp'] else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dict = 'trained_3c'

    opt = {
        'n_epochs': default_args['n_epochs'],
        'sequence_length': default_args['sequence_length'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'path_test': default_args['path_test'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'n_conditions': len(default_args['conditions']),
        'patch_size': default_args['patch_size'],
        'hidden_dim': 128,  # Dimension of hidden layers in discriminator and generator
        'world_size': world_size,  # number of processes for distributed training
        'device': device,
    }

    # TODO: implement data concatenation of experiment and generator data
    # Load dataset as tensor
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None

    # if in testing mode and path_test is None, use the dataset from the specified checkpoint
    if default_args['testing'] and default_args['path_test'] == 'None':
        default_args['path_test'] = default_args['path_checkpoint']
        opt['path_test'] = default_args['path_checkpoint']

    # Get test data if provided
    if default_args['path_test'] != 'None':
        if default_args['path_test'].endswith('.pt'):
            # load checkpoint and extract test_dataset
            test_data = torch.load(default_args['path_test'], map_location=device)['test_dataset'][:,
                        opt['n_conditions']:].float()
            test_labels = torch.load(default_args['path_test'], map_location=device)['test_dataset'][:,
                          :opt['n_conditions']].float()
        elif default_args['path_test'].endswith('.csv'):
            # load csv
            dataloader = Dataloader(default_args['path_test'],
                                    kw_timestep=default_args['kw_timestep_dataset'],
                                    col_label=default_args['conditions'],
                                    norm_data=True)
            test_data = dataloader.get_data()[:, opt['n_conditions']:].float()
            test_labels = dataloader.get_data()[:, :opt['n_conditions']].float()
            # test_data = torch.tensor(pd.read_csv(default_args['path_test']).to_numpy()[:, opt['n_conditions']:]).float()
            # test_labels = torch.tensor(pd.read_csv(default_args['path_test']).to_numpy()[:, :opt['n_conditions']]).float()

    if default_args['experiment']:
        # Get experiment's data as training data
        dataloader = Dataloader(default_args['path_dataset'],
                                kw_timestep=default_args['kw_timestep_dataset'],
                                col_label=default_args['conditions'],
                                norm_data=True)
        if test_data is None:
            train_idx, test_idx = dataloader.dataset_split(train_size=.8)

            train_data = dataloader.get_data()[train_idx][:, dataloader.labels.shape[1]:]
            train_labels = dataloader.get_data()[train_idx][:, :dataloader.labels.shape[1]]
            test_data = dataloader.get_data()[test_idx][:, dataloader.labels.shape[1]:]
            test_labels = dataloader.get_data()[test_idx][:, :dataloader.labels.shape[1]]
        else:
            train_data = dataloader.get_data()[:, dataloader.labels.shape[1]:].float()
            train_labels = dataloader.get_data()[:, :dataloader.labels.shape[1]].float()

    if default_args['generated']:
        # Get generated data as training data
        train_data = torch.tensor(pd.read_csv(default_args['path_dataset']).to_numpy()[:, opt['n_conditions']:]).float()
        train_labels = torch.tensor(pd.read_csv(default_args['path_dataset']).to_numpy()[:, :opt['n_conditions']]).float()
        if test_data is None:
            # Split train data into train and test
            train_idx, test_idx = Dataloader().dataset_split(train_data, train_size=.8)
            test_data = train_data[test_idx].view(train_data[test_idx].shape).float()
            test_labels = train_labels[test_idx].view(train_labels[test_idx].shape).float()
            train_data = train_data[train_idx].view(train_data[train_idx].shape).float()
            train_labels = train_labels[train_idx].view(train_labels[train_idx].shape).float()

    opt['sequence_length'] = test_data.shape[1]# - len(default_args['conditions'])

    if opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(
            f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
            f"The sequence is padded with zeros to fit the condition.")
        padding = 0
        while (opt['sequence_length'] + padding) % default_args['patch_size'] != 0:
            padding += 1
        opt['sequence_length'] += padding
        train_data = torch.cat((train_data, torch.zeros(train_data.shape[0], padding)), dim=-1)
        test_data = torch.cat((test_data, torch.zeros(test_data.shape[0], padding)), dim=-1)

    # Load model and optimizer
    # if not default_args['testing']:
    critic_configuration = torch.load(default_args['path_critic'], map_location='cpu')

    critic = TtsDiscriminator(seq_length=critic_configuration['configuration']['sequence_length'],
                              patch_size=critic_configuration['configuration']['patch_size'],
                              in_channels=1 + critic_configuration['configuration']['n_conditions'])
    critic.load_state_dict(critic_configuration['discriminator'])
    critic.eval()
    classifier = TtsClassifier(seq_length=opt['sequence_length'],
                               patch_size=opt['patch_size'],
                               n_classes=int(opt['n_conditions']),
                               in_channels=3,
                               softmax=True).to(device)

    # Test model
    if default_args['testing']:
        classifier.load_state_dict(torch.load(default_args['path_checkpoint'], map_location=device)['classifier'])
        trainer = Trainer(classifier, critic, opt)
        fake_labels = torch.where(test_labels == 0, 1, 0)
        scores = trainer.compute_scores(test_data, test_labels, fake_labels)
        test_data_temp = test_data.view(-1, 1, 1, test_data.shape[-1])
        scores = scores.view(-1, 2, 1, 1).repeat(1, 1, 1, test_data.shape[-1]).to(trainer.device)
        test_data_temp = torch.concat((test_data_temp, scores), dim=1).to(trainer.device)
        test_loss, test_acc = trainer.test(test_data_temp, test_labels)
        print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")
        exit()

    # Train model
    if default_args['ddp']:
        # DDP Training
        trainer = DDPTrainer(classifier, critic, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        mp.spawn(run,
                 args=(world_size, find_free_port(), default_args['ddp_backend'], trainer, train_data, train_labels, test_data, test_labels),
                 nprocs=world_size, join=True)
    else:
        # Regular training
        trainer = Trainer(classifier, critic, opt)
        if default_args['load_checkpoint']:
            trainer.load_checkpoint(default_args['path_checkpoint'])
        loss = trainer.train(train_data, train_labels, test_data, test_labels)

        # Save model
        path = 'trained_3c'
        filename = '3c_' + timestamp + '.pt'
        filename = os.path.join(path, filename)
        trainer.save_checkpoint(filename, torch.concat((test_labels, test_data), dim=1), loss)

        print("Classifier training finished.")
        print("Model states, losses and test dataset saved to file: "
              f"\n{filename}.")


