import warnings

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

    def _gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP"""

        batch_size = real_samples.size(0)
        device = real_samples.device

        # Generate random epsilon
        epsilon = torch.rand(batch_size, 1, 1, device=device, requires_grad=True)
        epsilon = epsilon.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
        # interpolated_samples = torch.autograd.Variable(interpolated_samples, requires_grad=True)

        # Calculate critic scores for interpolated samples
        critic_scores = discriminator(interpolated_samples)

        # Compute gradients of critic scores with respect to interpolated samples
        gradients = torch.autograd.grad(outputs=critic_scores,
                                        inputs=interpolated_samples,
                                        grad_outputs=torch.ones(critic_scores.size(), device=device),
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Calculate gradient penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gradient_penalty_weight

        return gradient_penalty
