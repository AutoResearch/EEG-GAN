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
        return -torch.mean(real) + torch.mean(fake)

    def generator(self, fake):
        return -torch.mean(fake)


class WassersteinGradientPenaltyLoss(WassersteinLoss):
    def __init__(self):
        super().__init__(wgan=False)
        self.lambda_gp = 0

    def set_lambda_gp(self, lambda_gp):
        self.lambda_gp = lambda_gp

    def discriminator(self, *args):
        real, fake, discriminator, real_images, fake_images = args
        return super().discriminator(real, fake) + self._gradient_penalty(discriminator, real_images, fake_images)

    def _gradient_penalty(self, discriminator, real_images, fake_images):
        """Calculates the gradient penalty for WGAN-GP"""

        # adjust dimensions of real_labels, fake_labels and eta to to match the dimensions of real_images
        # if real_labels.shape != fake_labels.shape:
        #     raise ValueError("real_labels and fake_labels must have the same shape!")

        if real_images.shape != fake_images.shape:
            raise ValueError("real_images and fake_images must have the same shape!")

        # check that all inputs' devices are the same
        if real_images.device != fake_images.device:
            raise ValueError("real_images and fake_images must be on the same device!")

        eta = torch.FloatTensor(real_images.shape[0], 1).uniform_(0, 1).repeat((1, real_images.shape[1])).to(real_images.device)

        # interpolate between real and fake images/labels
        # interpolated_labels = real_labels  # (eta * real_labels + ((1 - eta) * fake_labels))
        while eta.dim() < real_images.dim():
            eta = eta.unsqueeze(-1)
        interpolated = (eta * real_images + ((1 - eta) * fake_images))

        # concatenate interpolated and interpolated_labels along the channel dimension
        # repeat last dimension of interpolated_labels to match the last dimension of interpolated
        # if len(interpolated_labels.shape) == 2:
        #     interpolated_labels = interpolated_labels.unsqueeze(-1)
        # interpolated_labels = interpolated_labels.repeat(1, 1, interpolated.shape[-1])
        # while interpolated.dim() > interpolated_labels.dim():
            # keep 1st dim as batch_size; 2nd dim as channel_size; last dim as sequence_length
            # add as many dimensions between 2nd and last dim as necessary
            # interpolated_labels = interpolated_labels.unsqueeze(-2)
        # interpolated = torch.concat((interpolated, real_labels), dim=1)

        # define it to calculate gradient
        interpolated = autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated)

        fake = autograd.Variable(torch.ones((real_images.shape[0], 1)).to(real_images.device), requires_grad=False)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated,
                                  inputs=interpolated,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return grad_penalty
