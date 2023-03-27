import os
import torch
import torch.distributed as dist
from datetime import datetime, timedelta
from helpers.trainer_classifier import DDPTrainer


def run(rank, world_size, master_port, backend, trainer, train_data, train_labels, test_data, test_labels):
    _setup(rank, world_size, master_port, backend)
    trainer = _setup_trainer(rank, trainer)
    _ddp_training(trainer, train_data, train_labels, test_data, test_labels)
    dist.destroy_process_group()


def _setup(rank, world_size, master_port, backend):
    # print(f"Initializing process group on rank {rank}")# on master port {self.master_port}.")

    os.environ['MASTER_ADDR'] = 'localhost'  # '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)

    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=30))


def _setup_trainer(rank, trainer: DDPTrainer):
    # set device
    trainer.set_device(rank)
    print(f"Using device {trainer.device}.")

    # load checkpoint
    # if trainer.use_checkpoint:
    #     trainer.load_checkpoint(trainer.path_checkpoint)

    # construct DDP model
    trainer.set_ddp_framework()

    return trainer


def _ddp_training(trainer: DDPTrainer, train_data, train_labels, test_data, test_labels):
    # take partition of dataset for each process
    # start_index = int(len(train_data) / trainer.world_size * trainer.rank)
    # end_index = int(len(train_data) / trainer.world_size * (trainer.rank + 1))
    # train_data = train_data[start_index:end_index]
    # train_labels = train_labels[start_index:end_index]
    #
    # if trainer.batch_size > len(train_data):
    #     raise ValueError(f"Batch size {trainer.batch_size} is larger than the partition size {len(train_data)}.")

    # train
    loss = trainer.train(train_data, train_labels, test_data, test_labels)

    # save checkpoint
    if trainer.rank == 0:
        test_dataset = torch.concat((test_labels, test_data), dim=1)
        path = '../trained_classifier'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'classifier_ddp_{trainer.epochs}ep_' + timestamp + '.pt'
        trainer.save_checkpoint(os.path.join(path, filename), test_dataset, loss)

        print("Classifier training finished.")
        print("Model states, losses and test dataset saved to file: "
              f"\n{filename}.")