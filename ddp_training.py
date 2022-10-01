import os
import socket
import random
from datetime import datetime, timedelta

import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import trainer
from losses import WassersteinGradientPenaltyLoss as Loss
import losses
from get_master import find_free_port


class DDPTrainer(trainer.Trainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        super(trainer.Trainer, self).__init__()

        # training configuration
        super().__init__(generator, discriminator, opt)

        self.rank = None

        self.world_size = opt['world_size'] if 'world_size' in opt else 1

    # ---------------------
    #  DDP-specific modifications
    # ---------------------

    def save_checkpoint(self, path_checkpoint=None, generated_samples=None):
        if self.rank == 0:
            super().save_checkpoint(path_checkpoint, generated_samples)
        dist.barrier()

    def print_log(self, current_epoch, current_batch, num_batches, d_loss, g_loss):
        # average the loss across all processes before printing
        reduce_tensor = torch.tensor([d_loss, g_loss], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size

        super().print_log(current_epoch, current_batch, num_batches, reduce_tensor[0], reduce_tensor[1])

    def set_device(self, rank):
        self.rank = rank
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' + ':' + str(rank))

    def set_ddp_framework(self):
        # set ddp generator and discriminator
        self.generator.to(self.rank)
        self.discriminator.to(self.rank)
        self.generator = DDP(self.generator, device_ids=[self.rank])
        self.discriminator = DDP(self.discriminator, device_ids=[self.rank])

        # safe optimizer state_dicts for later use
        g_opt_state = self.generator_optimizer.state_dict()
        d_opt_state = self.discriminator_optimizer.state_dict()

        self.generator_optimizer = torch.optim.Adam(self.ddp_generator.parameters(),
                                                    lr=self.learning_rate, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.ddp_discriminator.parameters(),
                                                        lr=self.learning_rate, betas=(self.b1, self.b2))

        self.generator_optimizer.load_state_dict(g_opt_state)
        self.discriminator_optimizer.load_state_dict(d_opt_state)


def run(rank, world_size, master_port, training, dataset):
    _setup(rank, world_size, master_port)
    training = _setup_training(rank, training)
    _ddp_training(training, dataset)
    dist.destroy_process_group()


def _setup(rank, world_size, master_port):
    # print(f"Initializing process group on rank {rank}")# on master port {self.master_port}.")

    os.environ['MASTER_ADDR'] = 'localhost'#'127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)

    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))


def _setup_training(rank, training):
    # set device
    training.set_device(rank)
    print(f"Using device {training.device}.")

    # construct DDP model
    training.set_ddp_framework()

    return training


def _ddp_training(training: DDPTrainer, dataset):

    gen_samples = training.training(dataset)

    # save checkpoint
    if training.rank == 0:
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'state_dict_ddp_{training.epochs}ep_' + timestamp + '.pt'
        training.save_checkpoint(path_checkpoint=os.path.join(path, filename), generated_samples=gen_samples)

    print("GAN training finished.")
    print("Model states and generated samples saved to file.")
