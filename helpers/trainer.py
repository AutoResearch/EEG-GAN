import os

import torch
import numpy as np

from nn_architecture import losses, models
from nn_architecture.losses import WassersteinGradientPenaltyLoss as Loss

# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
# For implementation of Wasserstein-GAN see link above


class Trainer:
    """Trainer for conditional Wasserstein-GAN with gradient penalty.
    Source: https://arxiv.org/pdf/1704.00028.pdf"""

    def __init__(self, generator, discriminator, opt):
        # training configuration
        self.device = opt['device'] if 'device' in opt else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = opt['sequence_length'] if 'sequence_length' in opt else None
        self.sequence_length_generated = opt['seq_len_generated'] if 'seq_len_generated' in opt else None
        self.batch_size = opt['batch_size'] if 'batch_size' in opt else 32
        self.epochs = opt['n_epochs'] if 'n_epochs' in opt else 10
        self.use_checkpoint = opt['load_checkpoint'] if 'load_checkpoint' in opt else False
        self.path_checkpoint = opt['path_checkpoint'] if 'path_checkpoint' in opt else None
        self.latent_dim = opt['latent_dim'] if 'latent_dim' in opt else 10
        self.critic_iterations = opt['critic_iterations'] if 'critic_iterations' in opt else 5
        self.lambda_gp = opt['lambda_gp'] if 'lambda_gp' in opt else 10
        self.sample_interval = opt['sample_interval'] if 'sample_interval' in opt else 100
        self.learning_rate = opt['learning_rate'] if 'learning_rate' in opt else 0.0001
        self.n_conditions = opt['n_conditions'] if 'n_conditions' in opt else 0
        self.b1 = 0  # .5
        self.b2 = 0.9  # .999
        self.rank = 0  # Device: cuda:0, cuda:1, ... --> Device: cuda:rank

        self.generator = generator
        self.discriminator = discriminator

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr=self.learning_rate, betas=(self.b1, self.b2))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.learning_rate, betas=(self.b1, self.b2))

        self.loss = Loss()
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            self.loss.set_lambda_gp(self.lambda_gp)

        self.prev_g_loss = 0
        self.configuration = {
            'device': self.device,
            'generator': str(self.generator.__class__.__name__),
            'discriminator': str(self.discriminator.__class__.__name__),
            'sequence_length': self.sequence_length,
            'sequence_length_generated': self.sequence_length_generated,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'sample_interval': self.sample_interval,
            'learning_rate': self.learning_rate,
            'n_conditions': self.n_conditions,
            'latent_dim': self.latent_dim,
            'critic_iterations': self.critic_iterations,
            'lambda_gp': self.lambda_gp,
            'patch_size': opt['patch_size'] if 'patch_size' in opt else None,
            'b1': self.b1,
            'b2': self.b2,
            'path_dataset': opt['path_dataset'] if 'path_dataset' in opt else None,
        }

        self.d_losses = []
        self.g_losses = []

        # # load checkpoint
        # try:
        #     if self.use_checkpoint:
        #         self.load_checkpoint(self.path_checkpoint)
        #         self.use_checkpoint = False
        # except RuntimeError:
        #     Warning("Could not load checkpoint. If DDP was used while saving and is used now for loading the checkpoint will be loaded in a following step.")

    def training(self, dataset):
        """Batch training of the conditional Wasserstein-GAN with GP."""

        self.generator.train()
        self.discriminator.train()

        gen_samples = []
        num_batches = int(np.ceil(dataset.shape[0] / self.batch_size))

        # checkpoint file settings; toggle between two checkpoints to avoid corrupted file if training is interrupted
        path_checkpoint = 'trained_models'
        trigger_checkpoint_01 = True
        checkpoint_01_file = 'checkpoint_01.pt'
        checkpoint_02_file = 'checkpoint_02.pt'

        for epoch in range(self.epochs):
            # for-loop for number of batch_size entries in sessions
            dataset = dataset[torch.randperm(dataset.shape[0])]
            for i in range(0, dataset.shape[0], self.batch_size):
                # print(f'rank {self.rank} starts new batch...')
                # Check whether last batch contains less samples than batch_size
                if i + self.batch_size > dataset.shape[0]:
                    batch_size = dataset.shape[0] - i  # set batch_size to the remaining number of entries
                else:
                    batch_size = self.batch_size

                # draw batch_size samples from sessions
                data = dataset[i:i + batch_size, self.n_conditions:].to(self.device)
                data_labels = dataset[i:i + batch_size, :self.n_conditions].to(self.device)

                # update generator every n iterations as suggested in paper
                if int(i / batch_size) % self.critic_iterations == 0:
                    train_generator = True
                else:
                    train_generator = False

                d_loss, g_loss, gen_imgs = self.batch_train(data, data_labels, train_generator)

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

            # Save a checkpoint of the trained GAN and the generated samples every sample interval
            if epoch % self.sample_interval == 0:
                gen_samples.append(gen_imgs[np.random.randint(0, batch_size), :].detach().cpu().numpy())
                # save models and optimizer states as checkpoints
                # toggle between checkpoint files to avoid corrupted file during training
                if trigger_checkpoint_01:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_01_file), generated_samples=gen_samples)
                    trigger_checkpoint_01 = False
                else:
                    self.save_checkpoint(os.path.join(path_checkpoint, checkpoint_02_file), generated_samples=gen_samples)
                    trigger_checkpoint_01 = True

            self.print_log(epoch + 1, d_loss, g_loss)

        self.manage_checkpoints(path_checkpoint, [checkpoint_01_file, checkpoint_02_file])

        return gen_samples

    def batch_train(self, data, data_labels, train_generator):
        """Trains the GAN-Model on one batch of data.
        No further batch-processing. Give batch as to-be-used."""
        batch_size = data.shape[0]
        # TODO: for channel recovery: comment if-case out -> To get a 2D latent variable
        seq_length = data.shape[1] if isinstance(self.generator, models.CondLstmGenerator) else 1

        gen_cond_data = data[:, :self.sequence_length-self.sequence_length_generated].to(self.device)
        if train_generator:

            # -----------------
            #  Train Generator
            # -----------------

            self.generator_optimizer.zero_grad()

            # Sample noise and labels as generator input
            z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim,
                                            device=self.device, sequence_length=seq_length)
            gen_labels = torch.cat((data_labels, gen_cond_data), dim=1).to(self.device)

            # Generate a batch of samples
            # if isinstance(self.generator, models.TtsGenerator):
            # TODO: for channel recovery: concatenate z and ALL channels along dim=0
            z = torch.cat((z, gen_labels), dim=1)
            gen_imgs = self.generator(z)
            # TODO: for channel recovery: Output of G is 4-dim: (batch_size, n_channels, 1, seq_length)
            # else:
            #     gen_imgs = self.generator(z, gen_labels)

            # if isinstance(self.discriminator, models.TtsDiscriminator):
            # print devices of tensors in torch.cat
            # TODO: for channel recovery: Whenever tensor.view() Replace 2nd dim with n_channels
            # dim_labels = data_labels.shape = (batch_size, n_conditions)
            # dim_gen_imgs = gen_imgs.shape = (batch_size, n_channels, 1, seq_length)
            # Concatenate labels and images
            # torch.cat((data_labels, gen_imgs), dim=1) --> New shape: (batch_size, n_conditions + n_channels, 1, seq_length)
            fake_data = torch.cat((gen_cond_data.view(batch_size, 1, 1, -1), gen_imgs), dim=-1).to(self.device)
            fake_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length).to(self.device)
            fake_data = torch.cat((fake_data, fake_labels), dim=1).to(self.device)
            validity = self.discriminator(fake_data)
            # else:
            #     validity = self.discriminator(gen_imgs, gen_labels)

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

        # if isinstance(self.generator, models.TtsGenerator):
        # Sample noise and labels as generator input
        z = self.sample_latent_variable(batch_size=batch_size, latent_dim=self.latent_dim,
                                        device=self.device, sequence_length=seq_length)
        gen_labels = torch.cat((data_labels, gen_cond_data), dim=1).to(self.device)
        z = torch.cat((z, gen_labels), dim=1).to(self.device)
        gen_imgs = self.generator(z)

        # TODO: for channel recovery: Take only the fixed channels and replace the broken ones with the fixed ones
        # TODO: for channel recovery: Give the new matrix as input to D
        # TODO: for channel recovery: Additional information for D could be which channel was fixed -> twice as many labels for D
        # TODO: for channel recovery: Take cond data and reshape to 4D tensor: (batch_size, 1, n_channels, seq_length)

        gen_samples = torch.cat((gen_labels, gen_imgs.view(gen_imgs.shape[0], gen_imgs.shape[-1])), dim=1).to(self.device)

        # Loss for fake images
        fake_data = torch.cat((gen_cond_data.view(batch_size, 1, 1, -1), gen_imgs), dim=-1).to(self.device)
        fake_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length).to(self.device)
        fake_data = torch.cat((fake_data, fake_labels), dim=1).to(self.device)
        validity_fake = self.discriminator(fake_data)

        # Loss for real images
        real_labels = data_labels.view(-1, 1, 1, 1).repeat(1, 1, 1, self.sequence_length).to(self.device)
        data = data.view(-1, 1, 1, data.shape[1]).to(self.device)
        real_data = torch.cat((data, real_labels), dim=1).to(self.device)
        validity_real = self.discriminator(real_data)
        # else:
        #     # Loss for real images
        #     validity_real = self.discriminator(data, data_labels)
        #
        #     # Loss for fake images
        #     validity_fake = self.discriminator(gen_imgs, gen_labels)

        # Total discriminator loss and update
        if isinstance(self.loss, losses.WassersteinGradientPenaltyLoss):
            # discriminator, real_images, fake_images, real_labels, fake_labels
            d_loss = self.loss.discriminator(validity_real, validity_fake, self.discriminator, real_data, fake_data)
        else:
            d_loss = self.loss.discriminator(validity_real, validity_fake)
        d_loss.backward()
        self.discriminator_optimizer.step()
        # print(f'rank {self.rank}: Updated Discriminator')

        return d_loss.item(), g_loss, gen_samples

    def save_checkpoint(self, path_checkpoint=None, generated_samples=None, generator=None, discriminator=None):
        if path_checkpoint is None:
            path_checkpoint = 'trained_models'+os.path.sep+'checkpoint.pt'
        if generator is None:
            generator = self.generator
        if discriminator is None:
            discriminator = self.discriminator

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'discriminator_loss': self.d_losses,
            'generator_loss': self.g_losses,
            'generated_samples': generated_samples,
            'configuration': self.configuration,
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        if os.path.isfile(path_checkpoint):
            # load state_dicts
            state_dict = torch.load(path_checkpoint, map_location=self.device)
            self.generator.load_state_dict(state_dict['generator'])
            self.discriminator.load_state_dict(state_dict['discriminator'])
            self.generator_optimizer.load_state_dict(state_dict['generator_optimizer'])
            self.discriminator_optimizer.load_state_dict(state_dict['discriminator_optimizer'])
            print(f"Device {self.device}:{self.rank}: Using pretrained GAN.")
        else:
            Warning("No checkpoint-file found. Using random initialization.")

    def manage_checkpoints(self, path_checkpoint: str, checkpoint_files: list, generator=None, discriminator=None):
        """if training was successful delete the sub-checkpoint files and save the most current state as checkpoint,
        but without generated samples to keep memory usage low. Checkpoint should be used for further training only.
        Therefore, there's no need for the saved samples."""

        print("Managing checkpoints...")
        # save current model as checkpoint.pt
        self.save_checkpoint(path_checkpoint=os.path.join(path_checkpoint, 'checkpoint.pt'), generator=generator, discriminator=discriminator)

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
    def sample_latent_variable(sequence_length=1, batch_size=1, latent_dim=1, device=torch.device('cpu')):
        """samples a latent variable from a normal distribution
        as a tensor of shape (batch_size, (sequence_length), latent_dim) on the given device"""
        if sequence_length > 1:
            # sample a sequence of latent variables
            # only used for RNN/LSTM generator
            return torch.randn((batch_size, sequence_length, latent_dim), device=device).float()
        else:
            return torch.randn((batch_size, latent_dim), device=device).float()
