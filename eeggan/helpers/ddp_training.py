import os
from datetime import datetime, timedelta
import numpy as np

import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import eeggan.helpers.trainer as trainer
from eeggan.helpers.dataloader import Dataloader


class GANDDPTrainer(trainer.GANTrainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        super(trainer.GANTrainer, self).__init__()

        # training configuration
        super().__init__(generator, discriminator, opt)

        self.world_size = opt['world_size'] if 'world_size' in opt else 1

    # ---------------------
    #  DDP-specific modifications
    # ---------------------

    def save_checkpoint(self, path_checkpoint=None, samples=None, generator=None, discriminator=None, update_history=False):
        if self.rank == 0:
            super().save_checkpoint(path_checkpoint, samples, generator=self.generator.module, discriminator=self.discriminator.module, update_history=update_history)
        # dist.barrier()

    def print_log(self, current_epoch, d_loss, g_loss):
        # if self.rank == 0:
        # average the loss across all processes before printing
        reduce_tensor = torch.tensor([d_loss, g_loss], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size

        super().print_log(current_epoch, reduce_tensor[0], reduce_tensor[1])

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None, samples=None, update_history=False):
        if self.rank == 0:
            super().manage_checkpoints(path_checkpoint, checkpoint_files, generator=self.generator.module, discriminator=self.discriminator.module, samples=samples, update_history=update_history)

    def set_device(self, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else f'cpu:{rank}')

    def set_ddp_framework(self):
        # set ddp generator and discriminator
        self.generator.to(self.rank)
        self.discriminator.to(self.rank)
        self.generator = DDP(self.generator, device_ids=[self.rank], find_unused_parameters=False) #TODO: We suppressed a warning that not all outputs were being used by adding the find_unused... argument. Should check further to see if this is here appropriate.
        self.discriminator = DDP(self.discriminator, device_ids=[self.rank], find_unused_parameters=False) #TODO: We suppressed a warning that not all outputs were being used by adding the find_unused... argument. Should check further to see if this is here appropriate.

        # safe optimizer state_dicts for later use
        g_opt_state = self.generator_optimizer.state_dict()
        d_opt_state = self.discriminator_optimizer.state_dict()

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.g_lr, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.d_lr, betas=(self.b1, self.b2))


class AEDDPTrainer(trainer.AETrainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, model, opt):
        super(trainer.AETrainer, self).__init__()

        # training configuration
        super().__init__(model, opt)

        self.world_size = opt['world_size'] if 'world_size' in opt else 1

    # ---------------------
    #  DDP-specific modifications
    # ---------------------

    def save_checkpoint(self, path_checkpoint=None, model=None, update_history=False, samples=None):
        if self.rank == 0:
            super().save_checkpoint(path_checkpoint=path_checkpoint, model=model, update_history=update_history, samples=samples)
        # dist.barrier()

    def print_log(self, current_epoch, train_loss, test_loss):
        reduce_tensor = torch.tensor([train_loss, test_loss], dtype=torch.float32, device=self.device)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
        reduce_tensor /= self.world_size

        super().print_log(current_epoch, reduce_tensor[0], reduce_tensor[1])

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, model=None, update_history=False, samples=None):
        if self.rank == 0:
            super().manage_checkpoints(path_checkpoint, checkpoint_files, model=self.model.module, update_history=update_history, samples=samples)

    def set_device(self, rank):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else f'cpu:{rank}')

    def set_ddp_framework(self):
        # set ddp generator and discriminator
        self.model.to(self.device)
        self.model.device = self.device
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True) #TODO: We suppressed a warning that not all outputs were being used by adding the find_unused... argument. Should check further to see if this is here appropriate.

        # safe optimizer state_dicts, init new ddp optimizer and load state_dicts
        opt_state = self.optimizer.state_dict()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


def run(rank, world_size, master_port, backend, trainer_ddp, opt):
    try:
        _setup(rank, world_size, master_port, backend)
        trainer_ddp = _setup_trainer(rank, trainer_ddp)
        _ddp_training(trainer_ddp, opt)
        dist.destroy_process_group()
    except Exception as error:
        ValueError(f"Error in DDP training: {error}")
        dist.destroy_process_group()


def _setup(rank, world_size, master_port, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    # create default process group
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=30))


def _setup_trainer(rank, trainer_ddp):
    # set device
    trainer_ddp.set_device(rank)
    print(f"Using device {trainer_ddp.device}.")

    # construct DDP model
    trainer_ddp.set_ddp_framework()

    return trainer_ddp

def _ddp_training(trainer_ddp, opt):
    # load data
    if 'conditions' not in opt:
        opt['conditions'] = ['']
    if isinstance(trainer_ddp, GANDDPTrainer):
        dataloader = Dataloader(opt['data'],
                            kw_time=opt['kw_time'],
                            kw_conditions=opt['kw_conditions'],
                            norm_data=opt['norm_data'],
                            std_data=opt['std_data'],
                            diff_data=opt['diff_data'],
                            kw_channel=opt['kw_channel'])
    elif isinstance(trainer_ddp, AEDDPTrainer):
        dataloader = Dataloader(opt['data'],
                            kw_time=opt['kw_time'],
                            norm_data=opt['norm_data'],
                            std_data=opt['std_data'],
                            diff_data=opt['diff_data'],
                            kw_channel=opt['kw_channel'])
    else:
        raise ValueError(f"Trainer type {type(trainer_ddp)} not supported.")
    
    dataset = dataloader.get_data()
    opt['sequence_length'] = dataset.shape[2] - dataloader.labels.shape[2]

    if trainer_ddp.batch_size > len(dataset):
        raise ValueError(f"Batch size {trainer_ddp.batch_size} is larger than the partition size {len(dataset)}.")

    # train
    if isinstance(trainer_ddp, GANDDPTrainer):
        path = 'trained_models'
        model_prefix = 'gan'
        dataset = DataLoader(dataset, batch_size=trainer_ddp.batch_size, shuffle=True)
        gen_samples = trainer_ddp.training(dataset)
    elif isinstance(trainer_ddp, AEDDPTrainer):
        path = 'trained_ae'
        model_prefix = 'ae'
        train_data = dataset[:int(len(dataset) * opt['train_ratio'])]
        test_data = dataset[int(len(dataset) * opt['train_ratio']):]
        train_data = DataLoader(train_data, batch_size=trainer_ddp.batch_size, shuffle=True)
        test_data = DataLoader(test_data, batch_size=trainer_ddp.batch_size, shuffle=True)
        trainer_ddp.training(train_data, test_data)
    else:
        raise ValueError(f"Trainer type {type(trainer_ddp)} not supported.")

    # save checkpoint
    if trainer_ddp.rank == 0:

        # save final models, optimizer states, generated samples, losses and configuration as final result
        path = 'trained_models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if opt['save_name'] != '':
            # check if .pt extension is already included in the save_name
            if not opt['save_name'].endswith('.pt'):
                opt['save_name'] += '.pt'
            filename = opt['save_name']
        else:
            filename = f'{model_prefix}_ddp_{trainer_ddp.epochs}ep_' + timestamp + '.pt'

        if isinstance(trainer_ddp, GANDDPTrainer):
            trainer_ddp.save_checkpoint(path_checkpoint=os.path.join(path, filename), samples=gen_samples)
        elif isinstance(trainer_ddp, AEDDPTrainer):
            samples = []
            for batch in test_data:
                inputs = batch.float().to(trainer_ddp.model.device)
                outputs = trainer_ddp.model(inputs)
                samples.append(np.concatenate([inputs.unsqueeze(1).detach().cpu().numpy(), outputs.unsqueeze(1).detach().cpu().numpy()], axis=1))
            trainer_ddp.save_checkpoint(path_checkpoint=os.path.join(path, filename), samples=samples)

        print("Model training finished.")
        print(f"Model states and generated samples saved to file {os.path.join(path, filename)}.")