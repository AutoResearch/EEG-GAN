import os
import random
from datetime import datetime

import numpy as np
import torch

import models
import losses
from losses import WassersteinGradientPenaltyLoss as Loss

# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
# For implementation of Wasserstein-GAN see link above


class Trainer:
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt,
                 optimizer_generator=None, optimizer_discriminator=None, device=None):
        # training configuration
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # for distributed training
            self.device = device
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else None
        self.sequence_length_generated = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.critic_iterations = opt['critic_iterations'] if 'critic_iterations' in opt else 5
        self.lambda_gp = opt['lambda_gp'] if 'lambda_gp' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.b1 = 0  # .5
        self.b2 = 0.9  # .999

        self.generator = generator
        self.discriminator = discriminator

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.learning_rate, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.learning_rate, betas=(self.b1, self.b2))
        if optimizer_generator is not None:
            self.generator_optimizer.load_state_dict(optimizer_generator)
        if optimizer_discriminator is not None:
            self.discriminator_optimizer.load_state_dict(optimizer_discriminator)

        self.loss = Loss()
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            self.loss.set_lambda_gp(self.lambda_gp)

        self.prev_g_loss = 0

    def training(self, dataset):
        """Batch training of the conditional Wasserstein-GAN with GP."""

        gen_samples = []
        num_batches = int(np.ceil(dataset.shape[0] / self.batch_size))

        # checkpoint file settings; toggle between two checkpoints to avoid corrupted file if training is interrupted
        path_checkpoint = 'trained_models'
        checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'

        for epoch in range(self.epochs):
            # for-loop for number of batch_size entries in sessions
            random.shuffle(dataset)
            sessions = dataset[:, 1:]
            labels = dataset[:, 0].unsqueeze(-1)
            for i in range(0, len(sessions), self.batch_size):
                # Check whether last batch contains less samples than batch_size
                if i + self.batch_size > len(sessions):
                    batch_size = len(sessions) - i  # set batch_size to the remaining number of entries
                else:
                    batch_size = self.batch_size

                # draw batch_size samples from sessions
                data = sessions[i:i + batch_size]
                data_labels = labels[i:i + batch_size]

                # update generator every n iterations as suggested in paper
                if int(i / batch_size) % self.critic_iterations == 0:
                    train_generator = True
                else:
                    train_generator = False
                d_loss, g_loss, gen_imgs = self.batch_train(data, data_labels, train_generator)

                current_batch = i // self.batch_size + 1
                self.print_log(epoch+1, current_batch, num_batches, d_loss, g_loss)

                # Save a checkpoint of the trained GAN and the generated samples every sample interval
                batches_done = epoch * num_batches + current_batch
                if batches_done % self.sample_interval == 0:
                    gen_samples.append(gen_imgs[np.random.randint(0, batch_size), :].detach().cpu().numpy())
                    # save models and optimizer states as checkpoints
                    # toggle between checkpoint files to avoid corrupted file during training
                    if checkpoint_01:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), generated_samples=gen_samples)
                        checkpoint_01 = False
                    else:
                        self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), generated_samples=gen_samples)
                        checkpoint_01 = True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = r'trained_models\state_dict_'+timestamp+'.pt'
        self.save_checkpoint(path_checkpoint=path, generated_samples=gen_samples)

        return self.generator, self.discriminator, gen_samples

    def batch_train(self, data, data_labels, train_generator):
        """Trains the GAN-Model on one batch of data.
        No further batch-processing. Give batch as to-be-used."""
        batch_size = data.shape[0]
        seq_length = data.shape[1] if isinstance(self.generator, models.CondLstmGenerator) else 1

        print('\n-----------------------------\n')
        print(f'device is {self.device}')
        print('\n-----------------------------\n')

        data.to(self.device)
        data_labels.to(self.device)

        if train_generator:

            # -----------------
            #  Train Generator
            # -----------------

            self.generator_optimizer.zero_grad()

            # Sample noise and labels as generator input
            z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim,
                                            device=self.device, sequence_length=seq_length)
            # gen_labels = torch.randint(0, 2, (batch_size, 1)).to(self.device)
            gen_labels = torch.cat((data_labels, data[:, :self.sequence_length-self.sequence_length_generated]), dim=1).to(self.device)

            # Generate a batch of samples
            if isinstance(self.generator, models.TtsGenerator):
                z = torch.cat((z, gen_labels), dim=1)
                gen_imgs = self.generator(z)
            else:
                gen_imgs = self.generator(z, gen_labels)

            if isinstance(self.discriminator, models.TtsDiscriminator):
                cond_data = data[:, :self.sequence_length-self.sequence_length_generated].view(batch_size, 1, 1, -1)
                fake_data = torch.cat((cond_data, gen_imgs), dim=-1)
                fake_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length)
                fake_data = torch.cat((fake_data, fake_labels), dim=1)
                validity = self.discriminator(fake_data)
            else:
                validity = self.discriminator(gen_imgs, gen_labels)

            g_loss = self.loss.generator(validity)
            g_loss.backward()
            self.generator_optimizer.step()

            g_loss = g_loss.item()
            self.prev_g_loss = g_loss
        else:
            g_loss = self.prev_g_loss

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.discriminator_optimizer.zero_grad()

        if isinstance(self.generator, models.TtsGenerator):
            # Sample noise and labels as generator input
            z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim,
                                            device=self.device, sequence_length=seq_length)
            gen_labels = torch.cat((data_labels, data[:, :self.sequence_length - self.sequence_length_generated]), dim=1)
            z = torch.cat((z, gen_labels), dim=1)
            gen_imgs = self.generator(z)

            # Loss for fake images
            cond_data = data[:, :self.sequence_length - self.sequence_length_generated].view(batch_size, 1, 1, -1)
            fake_data = torch.cat((cond_data, gen_imgs), dim=-1)
            fake_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length)
            fake_data = torch.cat((fake_data, fake_labels), dim=1)
            validity_fake = self.discriminator(fake_data)

            # Loss for real images
            real_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length)
            data = data.view(-1, 1, 1, data.shape[1])
            real_data = torch.cat((data, real_labels), dim=1)
            validity_real = self.discriminator(real_data)
        else:
            # Loss for real images
            validity_real = self.discriminator(data, data_labels)

            # Loss for fake images
            validity_fake = self.discriminator(gen_imgs, gen_labels)

        # Total discriminator loss and update
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            d_loss = self.loss.discriminator(validity_real, validity_fake, self.discriminator, real_data, fake_data, data_labels, gen_labels)
        else:
            d_loss = self.loss.discriminator(validity_real, validity_fake)
        d_loss.backward()
        self.discriminator_optimizer.step()

        return d_loss.item(), g_loss, gen_imgs

    def save_checkpoint(self, path_checkpoint=None, generated_samples=None):
        if path_checkpoint is None:
            path_checkpoint = r'trained_models\checkpoint.pt'

        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'generated_samples': generated_samples,
        }, path_checkpoint)

    def print_log(self, current_epoch, current_batch, num_batches, d_loss, g_loss):
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (current_epoch, self.epochs,
               current_batch, num_batches,
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
    def sample_latent_variable(sequence_length=1, batch_size=1, latent_dim=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        if sequence_length > 1:
            # sample a sequence of latent variables
            # only used for RNN/LSTM generator
            return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()
        else:
            return torch.randn((batch_size, latent_dim), device=device).float()
