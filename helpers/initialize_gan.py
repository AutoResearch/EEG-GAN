import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder
from nn_architecture.models import TTSGenerator, TTSDiscriminator, DecoderGenerator, EncoderDiscriminator


gan_architectures = {
        'TTSGenerator': lambda seq_len, hidden_dim, patch_size, channels, latent_dim, num_layers, num_heads, **kwargs: TTSGenerator(seq_len, patch_size, channels, 1, latent_dim, 10, num_layers, num_heads, 0.5, 0.5),
        'TTSDiscriminator': lambda channels, hidden_dim, patch_size, seq_len, num_layers, **kwargs: TTSDiscriminator(channels, patch_size, 50, seq_len, num_layers, 1),
    }

gan_types = {
        'tts': ['TTSGenerator', 'TTSDiscriminator'],
    }


def init_gan(latent_dim_in, 
             channel_in_disc, 
             n_channels, 
             n_conditions,
             device,
             sequence_length_generated=-1,
             hidden_dim=128, 
             num_layers=2, 
             activation='tanh', 
             input_sequence_length=0, 
             patch_size=-1, 
             autoencoder='',
             **kwargs,
             ):
    if autoencoder == '':
        # no autoencoder defined -> use transformer GAN
        generator = gan_architectures[gan_types['tts'][0]](
            latent_dim=latent_dim_in,
            channels=n_channels,
            seq_len=sequence_length_generated,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            activation=activation,
            num_heads=4,

            # additional TTSGenerator inputs: patch_size
            patch_size=patch_size,
        )

        discriminator = gan_architectures[gan_types['tts'][1]](
            channels=channel_in_disc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            seq_len=sequence_length_generated,
            num_heads=4,

            # additional TTSDiscriminator inputs: patch_size
            patch_size=patch_size,
        )
    else:
        # initialize an autoencoder-GAN

        # initialize the autoencoder
        ae_dict = torch.load(autoencoder, map_location=torch.device('cpu'))
        if ae_dict['configuration']['target'] == 'channels':
            ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_CHANNELS
            autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
        elif ae_dict['configuration']['target'] == 'time':
            ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_TIMESERIES
            autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
        elif ae_dict['configuration']['target'] == 'full':
            autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=2).to(device)
            autoencoder.model_1 = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=1).to(device)
            autoencoder.model_1.eval()
        else:
            raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
        consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
        autoencoder.load_state_dict(ae_dict['model'])
        # freeze the autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()
        
        # adjust generator output_dim to match the output_dim of the autoencoder
        n_channels = autoencoder.output_dim if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim_2
        sequence_length_generated = autoencoder.output_dim_2 if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim

        # adjust discriminator input_dim to match the output_dim of the autoencoder
        channel_in_disc = n_channels + n_conditions

        generator = DecoderGenerator(
            generator=gan_architectures[gan_types['tts'][0]](
                latent_dim=latent_dim_in,
                channels=n_channels,
                seq_len=sequence_length_generated,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                activation=activation,
                num_heads=4,

                # additional TTSGenerator inputs: patch_size
                patch_size=patch_size,
            ),
            decoder=autoencoder
        )

        discriminator = EncoderDiscriminator(
            discriminator=gan_architectures[gan_types['tts'][1]](
                channels=channel_in_disc,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                seq_len=sequence_length_generated,
                num_heads=4,

                # additional TTSDiscriminator inputs: patch_size
                patch_size=patch_size,
            ),
            encoder=autoencoder
        )

        if isinstance(generator, DecoderGenerator) and isinstance(discriminator, EncoderDiscriminator) and input_sequence_length == 0:
            # if input_sequence_length is 0, do not decode the generator output during training
            generator.decode_output(False)

        if isinstance(discriminator, EncoderDiscriminator) and isinstance(generator, DecoderGenerator) and input_sequence_length == 0:
            # if input_sequence_length is 0, do not encode the discriminator input during training
            discriminator.encode_input(False)
            
    return generator, discriminator