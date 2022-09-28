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
        super().__init__(generator, discriminator, opt, device='cpu')

        self.ddp_generator = None
        self.ddp_discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    # ---------------------
    #  DDP-specific modifications
    # ---------------------

    def save_checkpoint(self, path_checkpoint=None, generated_samples=None):
        if self.device == 0:
            if path_checkpoint is None:
                path_checkpoint = r'trained_models\checkpoint.pt'

            torch.save({
                'generator': self.ddp_generator.state_dict(),
                'discriminator': self.ddp_discriminator.state_dict(),
                'generator_optimizer': self.generator_optimizer.state_dict(),
                'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
                'generated_samples': generated_samples,
            }, path_checkpoint)
        dist.barrier()

    def print_log(self, current_epoch, current_batch, num_batches, d_loss, g_loss):
        # average the loss across all processes before printing
        reduce_tensor = torch.tensor([d_loss, g_loss], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size

        super().print_log(current_epoch, current_batch, num_batches, reduce_tensor[0], reduce_tensor[1])


def run(rank, world_size, master_port, training, dataset, trained_gan=False):
    _setup(rank, world_size, master_port)
    training = _setup_training(rank, training, trained_gan)
    _ddp_training(rank, training, dataset)
    dist.destroy_process_group()


def _ddp_training(rank, training: DDPTrainer, dataset):

    _, _, gen_samples = training.training(dataset)

    # save checkpoint
    if rank == 0:
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'state_dict_ddp_{training.epochs}ep_' + timestamp + '.pt'
        training.save_checkpoint(path_checkpoint=os.path.join(path, filename), generated_samples=gen_samples)

    print("GAN training finished.")
    print("Model states and generated samples saved to file.")


def _setup(rank, world_size, master_port):
    # print(f"Initializing process group on rank {rank}")# on master port {self.master_port}.")

    os.environ['MASTER_ADDR'] = 'localhost'#'127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)

    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))


def _setup_training(rank, training, trained_gan=False):
    # set device
    training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training.device = torch.device(training.device + ':' + str(rank))
    print(f"Using device {training.device}.")
    # construct DDP model
    training.generator.to(rank)
    training.discriminator.to(rank)
    if trained_gan:
        state_dict = _ddp_load_checkpoint(training.generator, training.discriminator, training.generator_optimizer, training.discriminator_optimizer)
        training.generator = state_dict[0]
        training.discriminator = state_dict[1]
        training.generator_optimizer = state_dict[2]
        training.discriminator_optimizer = state_dict[3]

    training.generator = DDP(training.generator, device_ids=[rank])
    training.discriminator = DDP(training.discriminator, device_ids=[rank])

    # define optimizer
    training.generator_optimizer = torch.optim.Adam(training.generator.parameters(),
                                                   lr=training.learning_rate,
                                                   betas=(training.b1, training.b2))
    training.discriminator_optimizer = torch.optim.Adam(training.discriminator.parameters(),
                                                       lr=training.learning_rate,
                                                       betas=(training.b1, training.b2))


    return training


def _ddp_load_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer, rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    map_location = {device+':%d' % 0: device+':%d' % rank}
    # if file checkpoint exists, load it
    path_checkpoint = r'trained_models\checkpoint.pt'
    if os.path.isfile(path_checkpoint):
        generator.load_state_dict(torch.load(path_checkpoint, map_location=map_location)['generator'])
        discriminator.load_state_dict(torch.load(path_checkpoint, map_location=map_location)['discriminator'])
        generator_optimizer.load_state_dict(torch.load(path_checkpoint, map_location=map_location)['generator_optimizer'])
        discriminator_optimizer.load_state_dict(torch.load(path_checkpoint, map_location=map_location)['discriminator_optimizer'])

    return generator, discriminator, generator_optimizer, discriminator_optimizer
