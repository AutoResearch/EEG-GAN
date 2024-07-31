from torch import nn

from nn_architecture.ae_networks import Autoencoder
from nn_architecture.tts_gan_components import Generator as TTSGenerator_Org, Discriminator as TTSDiscriminator_Org

# insert here all different kinds of generators and discriminators
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, z):
        raise NotImplementedError


class TTSGenerator(TTSGenerator_Org):
    def __init__(self, seq_len=150, patch_size=15, channels=3, num_classes=9, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(TTSGenerator, self).__init__(seq_len, patch_size, channels, num_classes, latent_dim, embed_dim, depth,
                 num_heads, forward_drop_rate, attn_drop_rate)


class TTSDiscriminator(TTSDiscriminator_Org):
    def __init__(self, in_channels=3, patch_size=15, emb_size=50, seq_length=150, depth=3, n_classes=1, **kwargs):
        super(TTSDiscriminator, self).__init__(in_channels, patch_size, emb_size, seq_length, depth, n_classes, **kwargs)


class DecoderGenerator(Generator):
    """
    DecoderGenerator serves as a wrapper for a generator.
    It takes the output of the generator and passes it to a given decoder if the corresponding flag was set.
    Otherwise, it returns the output of the generator.
    """

    def __init__(self, generator: Generator, decoder: Autoencoder):
        """
        :param generator: generator model
        :param decoder: autoencoder model that has a decode method
        """
        super(DecoderGenerator, self).__init__()
        self.generator = generator
        self.decoder = decoder
        self.decode = True

        # add attributes from generator
        self.latent_dim = generator.latent_dim if hasattr(generator, 'latent_dim') else None
        self.channels = generator.channels if hasattr(generator, 'channels') else None
        self.seq_len = generator.seq_len if hasattr(generator, 'seq_len') else None

    def forward(self, data):
        if self.decode:
            return self.decoder.decode(self.generator(data))
        else:
            return self.generator(data)

    def decode_output(self, mode=True):
        self.decode = mode


class EncoderDiscriminator(Discriminator):
    """
    EncoderDiscriminator serves as a wrapper for a discriminator.
    It takes the input of the discriminator and passes it to a given encoder if the corresponding flag was set.
    Otherwise, it returns the output of the discriminator.
    """

    def __init__(self, discriminator: Discriminator, encoder: Autoencoder):
        """
        :param discriminator: discriminator model
        :param encoder: autoencoder model that has an encode method
        """
        super(EncoderDiscriminator, self).__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.encode = True

        # add attributes from discriminator
        self.channels = discriminator.channels if hasattr(discriminator, 'channels') else None
        self.n_classes = discriminator.n_classes if hasattr(discriminator, 'n_classes') else None

    def forward(self, data):
        if self.encode:
            return self.encoder.encode(self.discriminator(data))
        else:
            return self.discriminator(data)

    def encode_input(self, mode=True):
        self.encode = mode