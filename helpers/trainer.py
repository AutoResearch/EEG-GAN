from doctest import debug_script
import os
import time
from tqdm import tqdm
from decimal import Decimal
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

import nn_architecture.losses as losses
from nn_architecture.losses import WassersteinGradientPenaltyLoss as Loss
from nn_architecture.models import TransformerGenerator, TransformerDiscriminator, FFGenerator, FFDiscriminator, TTSGenerator, TTSDiscriminator, DecoderGenerator, EncoderDiscriminator


class Trainer:
    def __init__(self):
        pass

    def training(self):
        raise NotImplementedError

    def batch_train(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def manage_checkpoints(self):
        raise NotImplementedError

    def print_log(self):
        raise NotImplementedError


class GANTrainer(Trainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else 0
        self.input_sequence_length = opt['input_sequence_length'] if 'input_sequence_length' in opt else 0
        self.sequence_length_generated = self.sequence_length-self.input_sequence_length if self.sequence_length != self.input_sequence_length else self.sequence_length
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.use_checkpoint = opt['load_checkpoint'] if 'load_checkpoint' in opt else False
        self.path_checkpoint = opt['path_checkpoint'] if 'path_checkpoint' in opt else None
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.critic_iterations = opt['critic_iterations'] if 'critic_iterations' in opt else 5
        self.lambda_gp = opt['lambda_gp'] if 'lambda_gp' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.discriminator_lr = opt['discriminator_lr']
        self.generator_lr = opt['generator_lr']
        self.n_conditions = opt['n_conditions'] if 'n_conditions' in opt else 0
        self.n_channels = opt['n_channels'] if 'n_channels' in opt else 1
        self.channel_names = opt['channel_names'] if 'channel_names' in opt else list(range(0, self.n_channels))
        self.b1, self.b2 = 0, 0.9  # alternative values: .5, 0.999
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank
        self.lr_scheduler = opt['lr_scheduler']
        self.scheduler_warmup = opt['scheduler_warmup']
        self.scheduler_target = opt['scheduler_target']
        self.start_time = time.time()

        self.generator = generator
        self.discriminator = discriminator
        if hasattr(generator, 'module'):
            self.generator = {k.partition('module.')[2]: v for k,v in generator}
            self.discriminator = {k.partition('module.')[2]: v for k,v in discriminator}

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.d_lr = self.discriminator_lr if self.discriminator_lr is not None else self.learning_rate
        self.g_lr = self.generator_lr if self.generator_lr is not None else self.learning_rate

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.g_lr, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                lr=self.d_lr, betas=(self.b1, self.b2))
        
        self.generator_scheduler = None
        self.discriminator_scheduler = None
        if self.lr_scheduler.lower() == 'cycliclr':
            self.generator_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.generator_optimizer, base_lr=self.g_lr*.1, max_lr=self.g_lr, step_size_up=500, mode='exp_range', cycle_momentum=False, verbose=False)
            self.discriminator_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.discriminator_optimizer, base_lr=self.d_lr*.1, max_lr=self.d_lr, step_size_up=500, mode='exp_range', cycle_momentum=False, verbose=False)
        if self.scheduler_target.lower() == 'generator':
            self.discriminator_scheduler = None
        if self.scheduler_target.lower() == 'discriminator':
            self.generator_scheduler = None

        self.loss = Loss()
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            self.loss.set_lambda_gp(self.lambda_gp)

        self.d_losses = []
        self.g_losses = []
        self.trained_epochs = 0

        self.prev_g_loss = 0
        generator_class = str(self.generator.__class__.__name__) if not isinstance(self.generator, DecoderGenerator) else str(self.generator.generator.__class__.__name__)
        discriminator_class = str(self.discriminator.__class__.__name__) if not isinstance(self.discriminator, EncoderDiscriminator) else str(self.discriminator.discriminator.__class__.__name__)
        self.configuration = {
            'device': self.device,
            'generator_class': generator_class,
            'discriminator_class': discriminator_class,
            'sequence_length': self.sequence_length,
            'sequence_length_generated': self.sequence_length_generated,
            'input_sequence_length': self.input_sequence_length,
            'num_layers': opt['num_layers'],
            'hidden_dim': opt['hidden_dim'],
            'activation': opt['activation'],
            'latent_dim': self.latent_dim,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'trained_epochs': self.trained_epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'discriminator_lr': self.discriminator_lr,
            'generator_lr': self.generator_lr,
            'n_conditions': self.n_conditions,
            'latent_dim': self.latent_dim,
            'critic_iterations': self.critic_iterations,
            'lambda_gp': self.lambda_gp,
            'patch_size': opt['patch_size'] if 'patch_size' in opt else None,
            'b1': self.b1,
            'b2': self.b2,
            'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
            'path_autoencoder': opt['path_autoencoder'] if 'path_autoencoder' in opt else None,
            'n_channels': self.n_channels,
            'channel_names': self.channel_names,
            'lr_scheduler': self.lr_scheduler,
            'scheduler_warmup': self.scheduler_warmup,
            'scheduler_target': self.scheduler_target,
            'dataloader': {
                'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
                'column_label': opt['conditions'] if 'conditions' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_timestep': opt['kw_timestep'] if 'kw_timestep' in opt else None,
                'channel_label': opt['channel_label'] if 'channel_label' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else {},
        }

    def training(self, dataset: DataLoader):
        """Batch training of the conditional Wasserstein-GAN with GP."""
        gen_samples = []
        # checkpoint file settings; toggle between two checkpoints to avoid corrupted file if training is interrupted
        path_checkpoint = 'trained_models'
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        trigger_checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'

        gen_samples_batch = None
        batch = None

        loop = tqdm(range(self.epochs))
        for epoch in loop:
            # for-loop for number of batch_size entries in sessions
            i_batch = 0
            d_loss_batch = 0
            g_loss_batch = 0
            for batch in dataset:
                # draw batch_size samples from sessions
                data = batch[:, self.n_conditions:].to(self.device)
                data_labels = batch[:, :self.n_conditions, 0].unsqueeze(1).to(self.device)

                # update generator every n iterations as suggested in paper
                if i_batch % self.critic_iterations == 0:
                    train_generator = True
                else:
                    train_generator = False

                d_loss, g_loss, gen_samples_batch = self.batch_train(data, data_labels, train_generator)

                d_loss_batch += d_loss
                g_loss_batch += g_loss
                i_batch += 1
            self.d_losses.append(d_loss_batch/i_batch)
            self.g_losses.append(g_loss_batch/i_batch)

            if self.scheduler_warmup < epoch:
                if self.lr_scheduler.lower() == 'cycliclr':
                    if self.scheduler_target.lower() == 'generator' or self.scheduler_target.lower() == 'both':
                        self.generator_scheduler.step()
                    if self.scheduler_target.lower() == 'discriminator' or self.scheduler_target.lower() == 'both':
                        self.discriminator_scheduler.step()

            # Save a checkpoint of the trained GAN and the generated samples every sample interval
            if epoch % self.sample_interval == 0:
                gen_samples.append(gen_samples_batch[np.random.randint(0, len(batch))].detach().cpu().numpy())
                # save models and optimizer states as checkpoints
                # toggle between checkpoint files to avoid corrupted file during training
                if trigger_checkpoint_01:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=gen_samples)
                    trigger_checkpoint_01 = False
                else:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=gen_samples)
                    trigger_checkpoint_01 = True

            self.trained_epochs += 1
            #self.print_log(epoch + 1, d_loss_batch/i_batch, g_loss_batch/i_batch)
            loop.set_postfix_str(f"D LOSS: {np.round(d_loss_batch/i_batch,6)}, G LOSS: {np.round(g_loss_batch/i_batch,6)}")

        self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], samples=gen_samples, update_history=True)

        if isinstance(self.discriminator, EncoderDiscriminator):
            self.discriminator.encode_input()

        if isinstance(self.generator, DecoderGenerator):
            self.generator.decode_output()

        return gen_samples

    def batch_train(self, data, data_labels, train_generator):
        """Trains the GAN-Model on one batch of data.
        No further batch-processing. Give batch as to-be-used."""
        gen_cond_data_orig = None

        batch_size = data.shape[0]

        # gen_cond_data for prediction purposes; implemented but not tested right now;
        gen_cond_data = data[:, :self.input_sequence_length, :].to(self.device)

        # Channel recovery roughly implemented
        if self.input_sequence_length == self.sequence_length and self.n_channels > 1:
            recovery = 0.3
            zero_index = np.random.randint(0, self.n_channels, np.max((1, int(self.n_channels*recovery))))
            gen_cond_data[:, :, zero_index] = 0

        # if self.generator is instance of EncoderGenerator encode gen_cond_data to speed up training
        if isinstance(self.generator, DecoderGenerator) and self.input_sequence_length != 0:
            # pad gen_cond_data to match input sequence length of Encoder
            gen_cond_data_orig = gen_cond_data
            gen_cond_data = torch.cat((torch.zeros((batch_size, self.sequence_length - self.input_sequence_length, self.n_channels)).to(self.device), gen_cond_data), dim=1)
            gen_cond_data = self.generator.decoder.decode(gen_cond_data)

        seq_length = max(1, gen_cond_data.shape[1])
        gen_labels = torch.cat((gen_cond_data, data_labels.repeat(1, seq_length, 1).to(self.device)), dim=-1).to(self.device) if self.input_sequence_length != 0 else data_labels
        disc_labels = data_labels

        # -----------------
        #  Train Generator
        # -----------------
        if train_generator:

            # enable training mode for generator; disable training mode for discriminator + freeze discriminator weights
            self.generator.train()
            self.discriminator.eval()

            # Sample noise and labels as generator input
            z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim, sequence_length=seq_length, device=self.device)
            z = torch.cat((gen_labels, z), dim=-1).to(self.device)

            # Generate a batch of samples
            gen_imgs = self.generator(z)

            if gen_cond_data_orig is not None:
                # gen_cond_data was encoded before; use the original form to make fake data
                fake_data = self.make_fake_data(gen_imgs, data_labels, gen_cond_data_orig)
            else:
                fake_data = self.make_fake_data(gen_imgs, data_labels, gen_cond_data)

            # Compute loss/validity of generated data and update generator
            validity = self.discriminator(fake_data)
            g_loss = self.loss.generator(validity)
            self.generator_optimizer.zero_grad()
            g_loss.backward()
            self.generator_optimizer.step()

            g_loss = g_loss.item()
            self.prev_g_loss = g_loss
        else:
            g_loss = self.prev_g_loss

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # enable training mode for discriminator; disable training mode for generator + freeze generator weights
        self.generator.eval()
        self.discriminator.train()

        # Create a batch of generated samples
        with torch.no_grad():
            # Sample noise and labels as generator input
            z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim, sequence_length=seq_length, device=self.device)
            z = torch.cat((gen_labels, z), dim=-1).to(self.device)

            # Generate a batch of fake samples
            gen_imgs = self.generator(z)
            if gen_cond_data_orig is not None:
                # gen_cond_data was encoded before; use the original form to make fake data
                fake_data = self.make_fake_data(gen_imgs, disc_labels, gen_cond_data_orig)
            else:
                fake_data = self.make_fake_data(gen_imgs, disc_labels, gen_cond_data)

            if self.trained_epochs % self.sample_interval == 0:
                # decode gen_imgs if necessary - decoding only necessary if not prediction case or seq2seq case
                if not hasattr(self.generator, 'module'):
                    decode_imgs = isinstance(self.generator, DecoderGenerator) and not self.generator.decode
                else:
                    decode_imgs = isinstance(self.generator.module, DecoderGenerator) and not self.generator.module.decode

                if decode_imgs:
                    if not hasattr(self.generator, 'module'):
                        gen_samples = self.generator.decoder.decode(fake_data[:, :, :self.generator.channels].reshape(-1, self.generator.seq_len, self.generator.channels))
                    else:
                        gen_samples = self.generator.module.decoder.decode(fake_data[:, :, :self.generator.module.channels].reshape(-1, self.generator.module.seq_len, self.generator.module.channels))
                    # concatenate gen_cond_data_orig with decoded fake_data
                    # currently redundant because gen_cond_data is None in this case
                    if self.input_sequence_length != 0 and self.input_sequence_length != self.sequence_length:
                        if gen_cond_data_orig is not None:
                            gen_samples = torch.cat((gen_cond_data_orig, gen_samples), dim=1).to(self.device)
                        else:
                            gen_samples = torch.cat((gen_cond_data, gen_samples), dim=1).to(self.device)
                else:
                    gen_samples = fake_data[:, :, :self.n_channels]
                # concatenate channel names, conditions and generated samples
                gen_samples = torch.cat((data_labels.permute(0, 2, 1).repeat(1, 1, self.n_channels), gen_samples), dim=1)  # if self.n_conditions > 0 else gen_samples
            else:
                gen_samples = None

            if not hasattr(self.generator, 'module'):
                real_data = self.discriminator.encoder.encode(data) if isinstance(self.discriminator, EncoderDiscriminator) and not self.discriminator.encode else data
            else:
                real_data = self.discriminator.module.encoder.encode(data) if isinstance(self.discriminator.module, EncoderDiscriminator) and not self.discriminator.module.encode else data

            real_data = self.make_fake_data(real_data, disc_labels)

        # Loss for real and generated samples
        validity_fake = self.discriminator(fake_data)
        validity_real = self.discriminator(real_data)

        # Total discriminator loss and update
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            d_loss = self.loss.discriminator(validity_real, validity_fake, self.discriminator, real_data, fake_data)
        else:
            d_loss = self.loss.discriminator(validity_real, validity_fake)
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        return d_loss.item(), g_loss, gen_samples

    def save_checkpoint(self, path_checkpoint=None, samples=None, generator=None, discriminator=None, update_history=False):
        if path_checkpoint is None:
            path_checkpoint = 'trained_models'+os.path.sep+'checkpoint.pt'
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = [self.trained_epochs]
            self.configuration['train_time'] = time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))

        state_dict = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'generator_scheduler': None if self.generator_scheduler is None else self.generator_scheduler.state_dict(),
            'discriminator_scheduler': None if self.discriminator_scheduler is None else self.discriminator_scheduler.state_dict(),
            'generator_loss': self.g_losses,
            'discriminator_loss': self.d_losses,
            'samples': samples,
            'trained_epochs': self.trained_epochs,
            'configuration': self.configuration,
        }
        torch.save(state_dict, path_checkpoint)

        if update_history:
            print(f"Checkpoint saved to {path_checkpoint}.")
            print(f"Training complete in: {self.configuration['train_time']}")

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.generator_optimizer.load_state_dict(state_dict['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])

            '''
            self.generator_scheduler = None
            self.discriminator_scheduler = None
            if self.lr_scheduler.lower() == 'cycliclr':
                self.generator_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.generator_optimizer, base_lr=self.g_lr*.1, max_lr=self.g_lr, step_size_up=500, mode='exp_range', cycle_momentum=False, verbose=False)
                self.discriminator_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.discriminator_optimizer, base_lr=self.d_lr*.1, max_lr=self.d_lr, step_size_up=500, mode='exp_range', cycle_momentum=False, verbose=False)
            elif self.lr_scheduler.lower() == 'reducelronplateau':
                self.generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.generator_optimizer, factor=0.1, cooldown=50, verbose=False)
                self.discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.discriminator_optimizer, factor=0.1, cooldown=50, verbose=False)    
            if self.scheduler_target.lower() == 'generator':
                self.discriminator_scheduler = None
            if self.scheduler_target.lower() == 'discriminator':
                self.generator_scheduler = None
            '''

            if self.lr_scheduler != '' and state_dict['configuration']['lr_scheduler'] != '':
                if self.scheduler_target == 'generator' or self.scheduler_target == 'both':
                    self.generator_scheduler.load_state_dict(state_dict['generator_scheduler'])
                    for i in range(len(self.generator_optimizer.param_groups)):
                        self.generator_optimizer.param_groups[i]['lr'] = self.generator_scheduler.get_last_lr()[0]

                if self.scheduler_target == 'discriminator' or self.scheduler_target == 'both':
                    self.discriminator_scheduler.load_state_dict(state_dict['discriminator_scheduler'])
                    for i in range(len(self.generator_optimizer.param_groups)):
                        self.discriminator_optimizer.param_groups[i]['lr'] = self.discriminator_scheduler.get_last_lr()[0]
                        
            print(f"Device {self.device}:{self.rank}: Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. Using random initialization.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None, samples=None, update_history=False):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), generator=generator, discriminator=discriminator, samples=samples, update_history=update_history)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, d_loss, g_loss):
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (current_epoch, self.epochs,
               d_loss, g_loss)
        )

    def set_optimizer_state(self, optimizer, g_or_d='G'):
        if g_or_d == 'G':
            self.generator_optimizer.load_state_dict(optimizer)
            print('Generator optimizer state loaded successfully.')
        elif g_or_d == 'D':
            self.discriminator_optimizer.load_state_dict(optimizer)
            print('Discriminator optimizer state loaded successfully.')
        else:
            raise ValueError('G_or_D must be either "G" (Generator) or "D" (Discriminator)')

    @staticmethod
    def sample_latent_variable(batch_size=1, latent_dim=1, sequence_length=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()

    def make_fake_data(self, gen_imgs, data_labels, condition_data=None):
        """
        :param gen_imgs: generated data from generator of shape (batch_size, sequence_length, n_channels)
        :param data_labels: scalar labels/conditions for generated data of shape (batch_size, 1, n_conditions)
        :param condition_data: additional data for conditioning the generator of shape (batch_size, input_sequence_length, n_channels)
        """
        # if input_sequence_length is available and self.generator is instance of DecoderGenerator
        # decode gen_imgs, concatenate with gen_cond_data_orig and encode again to speed up training
        if self.input_sequence_length != 0 and self.input_sequence_length != self.sequence_length:
            # prediction case
            if gen_imgs.shape[1] == self.sequence_length:
                gen_imgs = gen_imgs[:, self.input_sequence_length:, :]
            fake_data = torch.cat((condition_data, gen_imgs), dim=1).to(self.device)
        else:
            fake_data = gen_imgs
        if data_labels.shape[-1] != 0:
            # concatenate labels with fake data if labels are available
            fake_data = torch.cat((fake_data, data_labels.repeat(1, fake_data.shape[1], 1)), dim=-1).to(self.device)

        return fake_data


class AETrainer(Trainer):
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, model, opt):
        # training configuration
        super().__init__()
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank

        # model
        self.model = model
        self.model.to(self.device)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()

        # training statistics
        self.trained_epochs = 0
        self.train_loss = []
        self.test_loss = []

        self.configuration = {
            'device': self.device,
            'model_class': str(self.model.__class__.__name__),
            'batch_size': self.batch_size,
            'n_epochs': self.epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'hidden_dim': opt['hidden_dim'],
            'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
            'path_checkpoint': opt['path_checkpoint'] if 'path_checkpoint' in opt else None,
            'timeseries_out': opt['timeseries_out'] if 'timeseries_out' in opt else None,
            'channels_out': opt['channels_out'] if 'channels_out' in opt else None,
            'target': opt['target'] if 'target' in opt else None,
            # 'conditions': opt['conditions'] if 'conditions' in opt else None,
            'channel_label': opt['channel_label'] if 'channel_label' in opt else None,
            'trained_epochs': self.trained_epochs,
            'input_dim': opt['input_dim'],
            'output_dim': opt['output_dim'],
            'output_dim_2': opt['output_dim_2'],
            'num_layers': opt['num_layers'],
            'num_heads': opt['num_heads'],
            'activation': opt['activation'],
            'dataloader': {
                'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
                'col_label': opt['conditions'] if 'conditions' in opt else None,
                'diff_data': opt['diff_data'] if 'diff_data' in opt else None,
                'std_data': opt['std_data'] if 'std_data' in opt else None,
                'norm_data': opt['norm_data'] if 'norm_data' in opt else None,
                'kw_timestep': opt['kw_timestep'] if 'kw_timestep' in opt else None,
                'channel_label': opt['channel_label'] if 'channel_label' in opt else None,
            },
            'history': opt['history'] if 'history' in opt else None,
        }

    def training(self, train_data, test_data):
        try:
            path_checkpoint = 'trained_ae'
            if not os.path.exists(path_checkpoint):
                os.makedirs(path_checkpoint)
            trigger_checkpoint_01 = True
            checkpoint_01_file = 'checkpoint_01.pt'
            checkpoint_02_file = 'checkpoint_02.pt'

            samples = []

            loop = tqdm(range(self.epochs))
            for epoch in loop:
                train_loss, test_loss, sample = self.batch_train(train_data, test_data)
                self.train_loss.append(train_loss)
                self.test_loss.append(test_loss)

                loop.set_postfix_str(f"TRAIN LOSS: {np.round(train_loss,6)}, TEST LOSS: {np.round(test_loss,6)}")

                if len(sample) > 0:
                    samples.append(sample)

                # Save a checkpoint of the trained GAN and the generated samples every sample interval
                if epoch % self.sample_interval == 0:
                    # save models and optimizer states as checkpoints
                    # toggle between checkpoint files to avoid corrupted file during training
                    if trigger_checkpoint_01:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), samples=samples)
                        trigger_checkpoint_01 = False
                    else:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), samples=samples)
                        trigger_checkpoint_01 = True

                self.trained_epochs += 1
                #self.print_log(epoch + 1, train_loss, test_loss)

            self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file], update_history=True, samples=samples)
            return samples

        except KeyboardInterrupt:
            # save model at KeyboardInterrupt
            print("Keyboard interrupt detected.\nSaving checkpoint...")
            self.save_checkpoint(update_history=True, samples=samples)

    def batch_train(self, train_data, test_data):
        train_loss = self.train_model(train_data)
        test_loss, samples = self.test_model(test_data)
        return train_loss, test_loss, samples

    def train_model(self, data):
        self.model.train()
        total_loss = 0
        for batch in data:
            self.optimizer.zero_grad()
            # inputs = nn.BatchNorm1d(batch.shape[-1])(batch.float().permute(0, 2, 1)).permute(0, 2, 1)
            # inputs = filter(inputs.detach().cpu().numpy(), win_len=random.randint(29, 50), dtype=torch.Tensor)
            inputs = batch.float().to(self.model.device)
            outputs = self.model(inputs)
            loss = self.loss(outputs, inputs)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data)

    def test_model(self, data):
        self.model.eval()
        total_loss = 0
        samples = []
        with torch.no_grad():
            for batch in data:
                inputs = batch.float().to(self.model.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, inputs)
                total_loss += loss.item()
                if self.trained_epochs % self.sample_interval == 0:
                    samples.append(np.concatenate([inputs.unsqueeze(1).detach().cpu().numpy(), outputs.unsqueeze(1).detach().cpu().numpy()], axis=1))
        if len(samples) > 0:
            samples = np.concatenate(samples, axis=0)[np.random.randint(0, len(samples))]
        return total_loss / len(data), samples

    def save_checkpoint(self, path_checkpoint=None, model=None, update_history=False, samples=None):
        if path_checkpoint is None:
            default_path = 'trained_ae'
            if not os.path.exists(default_path):
                os.makedirs(default_path)
            path_checkpoint = os.path.join(default_path, 'checkpoint.pt')

        if model is None:
            model = self.model

        if update_history:
            self.configuration['trained_epochs'] = self.trained_epochs
            self.configuration['history']['trained_epochs'] = self.configuration['history']['trained_epochs'] + [self.trained_epochs]
        
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'trained_epochs': self.trained_epochs,
            'samples': samples,
            'configuration': self.configuration,
        }

        torch.save(checkpoint_dict, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint-file {path_checkpoint} was not found.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, model=None, update_history=False, samples=None):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), model=None, update_history=update_history, samples=samples)

        for f in checkpoint_files:
            if os.path.exists(os.path.join(path_checkpoint, f)):
                os.remove(os.path.join(path_checkpoint, f))

    def print_log(self, current_epoch, train_loss, test_loss):
        print(
            "[Epoch %d/%d] [Train loss: %f] [Test loss: %f]" % (current_epoch, self.epochs, train_loss, test_loss)
        )

    def set_optimizer_state(self, optimizer):
        self.optimizer.load_state_dict(optimizer)
        print('Optimizer state loaded successfully.')
