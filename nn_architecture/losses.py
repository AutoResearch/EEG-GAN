import warnings
import numpy as np

import torch
from torch import autograd
# Implementation of the different loss functions


class Loss:
    def __init__(self):
        self.discriminator_loss = None
        self.generator_loss = None

    def discriminator(self, *args):
        pass

    def generator(self, *args):
        pass


class ConventionalLoss(Loss):
    def __init__(self):
        super().__init__()
        self.discriminator_loss = torch.nn.MSELoss()
        self.generator_loss = torch.nn.MSELoss()

    def discriminator(self, real, fake):
        real_loss = self.discriminator_loss(real, torch.ones_like(real))
        fake_loss = self.discriminator_loss(fake, torch.zeros_like(fake))
        return (real_loss + fake_loss) / 2

    def generator(self, validity_fake, fake=None):
        if fake is None:
            fake = torch.ones_like(validity_fake, dtype=torch.float32)
        return self.generator_loss(validity_fake, fake)


class WassersteinLoss(Loss):
    def __init__(self, wgan=True):
        super().__init__()
        self.discriminator_loss = None
        self.generator_loss = None

        if wgan:
            warnings.warn("No gradient clipping implemented for Wasserstein loss yet. "
                          "Please use WassersteinGradientPenaltyLoss if possible.")

    def discriminator(self, real, fake):
        # real and fake are the discriminator scores for real and fake samples;
        # the higher the score, the more realistic the sample looks to the discriminator
        # since the discriminator has to maximize its score for real samples and minimize it for fake samples,
        # the discriminator loss is the negative mean of the real and positive mean of the fake scores
        return -torch.mean(real) + torch.mean(fake)

    def generator(self, fake):
        # the generator has to maximize the discriminator score for fake samples,
        # so the generator loss is the negative mean of the fake scores
        return -torch.mean(fake)


class WassersteinGradientPenaltyLoss(WassersteinLoss):
    def __init__(self):
        super().__init__(wgan=False)
        self.gradient_penalty_weight = 0

    def set_lambda_gp(self, lambda_gp):
        self.gradient_penalty_weight = lambda_gp

    def discriminator(self, *args):
        real, fake, discriminator, real_images, fake_images = args
        return super().discriminator(real, fake) + self._gradient_penalty(discriminator, real_images, fake_images)

    def _gradient_penalty(self, discriminator: torch.nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor):
        """Calculates the gradient penalty for WGAN-GP"""
        
        # adjust dimensions of real_labels, fake_labels and eta to to match the dimensions of real_images
        # if real_labels.shape != fake_labels.shape:
        #     raise ValueError("real_labels and fake_labels must have the same shape!")

        if real_images.shape != fake_images.shape:
            raise ValueError("real_images and fake_images must have the same shape!")

        # check that all inputs' devices are the same
        if real_images.device != fake_images.device:
            raise ValueError("real_images and fake_images must be on the same device!")

        # Check that the inputs are of the dimension (batch_size, channels, 1, sequence_length)
        if real_images.dim() != 4:
            if real_images.dim() == 3:
                real_images, fake_images = real_images.unsqueeze(2), fake_images.unsqueeze(2)
            else:
                raise ValueError("real_images must be of dimension (batch_size, sequence_length, channels)!")
            real_images, fake_images = real_images.permute(0, 3, 2, 1), fake_images.permute(0, 3, 2, 1)
        
        eta = torch.FloatTensor(np.random.random((real_images.shape[0], 1))).to(real_images.device)#.uniform_(0, 1).repeat((1, real_images.shape[1]))

        # interpolate between real and fake images/labels
        # interpolated_labels = real_labels  # (eta * real_labels + ((1 - eta) * fake_labels))
        while eta.dim() < real_images.dim():
            eta = eta.unsqueeze(-1)
        interpolated = (eta * real_images.detach() + ((1 - eta) * fake_images.detach()))
        interpolated.requires_grad = True

        # deprecated - define it to calculate gradient
        # interpolated = autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)
        
        # fake = autograd.Variable(torch.ones((real_images.shape[0], 1)).to(real_images.device), requires_grad=False)
        fake = torch.ones((real_images.shape[0], 1), requires_grad=False).to(real_images.device)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated,
                                  inputs=interpolated,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gradient_penalty_weight
        return grad_penalty
    
    # TODO: Check why this gradient penalty is not working
    # def _gradient_penalty(self, discriminator, real_samples, fake_samples):
    #     """Calculates the gradient penalty for WGAN-GP"""

    #     batch_size = real_samples.size(0)
    #     device = real_samples.device

    #     # Generate random epsilon
    #     # TODO: this is old version
    #     # epsilon = torch.rand(batch_size, 1, 1, device=device, requires_grad=True)
    #     # epsilon = epsilon.expand_as(real_samples)
    #     # TODO: this is new version - replace old version with this
    #     # epsilon = torch.rand(*real_samples.shape, device=device, requires_grad=True)
    #     # TODO: this is the original pip version; delete if supposed to
    #     epsilon = torch.FloatTensor(real_samples.shape[0], 1).uniform_(0, 1).repeat((1, real_samples.shape[-1])).to(real_samples.device)
    #     while epsilon.dim() < real_samples.dim():
    #         epsilon = epsilon.unsqueeze(1)
            
    #     # Interpolate between real and fake samples
    #     interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
    #     interpolated_samples = torch.autograd.Variable(interpolated_samples, requires_grad=True)

    #     # Calculate critic scores for interpolated samples
    #     critic_scores = discriminator(interpolated_samples)

    #     # TODO: check out if fake works with new version (commented out) or only with old one
    #     # fake = torch.ones(critic_scores.size(), device=device)
    #     fake = autograd.Variable(torch.ones((real_samples.shape[0], 1)).to(real_samples.device), requires_grad=False)
    #     while fake.dim() < critic_scores.dim():
    #         fake = fake.unsqueeze(-1)
        
    #     # Compute gradients of critic scores with respect to interpolated samples
    #     gradients = torch.autograd.grad(outputs=critic_scores,
    #                                     inputs=interpolated_samples,
    #                                     grad_outputs=fake,
    #                                     create_graph=True,
    #                                     retain_graph=True)[0]

    #     # Calculate gradient penalty
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gradient_penalty_weight

    #     return gradient_penalty
