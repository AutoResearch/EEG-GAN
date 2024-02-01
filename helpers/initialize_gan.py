import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder
from nn_architecture.models import FFGenerator, FFDiscriminator, TransformerGenerator, TransformerDiscriminator, TTSGenerator, TTSDiscriminator, DecoderGenerator, EncoderDiscriminator


gan_architectures = {
        'FFGenerator': lambda latent_dim, channels, seq_len, hidden_dim, num_layers, dropout, activation, **kwargs: FFGenerator(latent_dim, channels, seq_len, hidden_dim, num_layers, dropout, activation),
        'FFDiscriminator': lambda channels, seq_len, hidden_dim, num_layers, dropout, **kwargs: FFDiscriminator(channels, seq_len, hidden_dim, num_layers, dropout),
        'TransformerGenerator': lambda latent_dim, channels, seq_len, hidden_dim, num_layers, num_heads, dropout, **kwargs: TransformerGenerator(latent_dim, channels, seq_len, hidden_dim, num_layers, num_heads, dropout),
        'TransformerDiscriminator': lambda channels, seq_len, hidden_dim, num_layers, num_heads, dropout, **kwargs: TransformerDiscriminator(channels, seq_len, 1, hidden_dim, num_layers, num_heads, dropout),
        'TTSGenerator': lambda seq_len, hidden_dim, patch_size, channels, latent_dim, num_layers, num_heads, **kwargs: TTSGenerator(seq_len, patch_size, channels, 1, latent_dim, 10, num_layers, num_heads, 0.5, 0.5),
        'TTSDiscriminator': lambda channels, hidden_dim, patch_size, seq_len, num_layers, **kwargs: TTSDiscriminator(channels, patch_size, 50, seq_len, num_layers, 1),
    }

gan_types = {
        'ff': ['FFGenerator', 'FFDiscriminator'],
        'tr': ['TransformerGenerator', 'TransformerDiscriminator'],
        'tts': ['TTSGenerator', 'TTSDiscriminator'],
    }


def init_gan(gan_type, 
             latent_dim_in, 
             channel_in_disc, 
             n_channels, 
             n_conditions,
             device,
             sequence_length_generated=-1,
             ae_sequence_length=-1,
             hidden_dim=128, 
             num_layers=2, 
             activation='tanh', 
             input_sequence_length=0, 
             patch_size=-1, 
             path_autoencoder='',
             **kwargs,
             ):
    if path_autoencoder == '':
        # no autoencoder defined -> use transformer GAN
        generator = gan_architectures[gan_types[gan_type][0]](
            # FFGenerator inputs: latent_dim, channels, hidden_dim, num_layers, dropout, activation
            latent_dim=latent_dim_in,
            channels=n_channels,
            seq_len=sequence_length_generated,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            activation=activation,

            # additional TransformerGenerator inputs: num_heads
            num_heads=4,

            # additional TTSGenerator inputs: patch_size
            patch_size=patch_size,
        )

        discriminator = gan_architectures[gan_types[gan_type][1]](
            # FFDiscriminator inputs: input_dim, hidden_dim, num_layers, dropout
            channels=channel_in_disc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            seq_len=sequence_length_generated,

            # TransformerDiscriminator inputs: channels, n_classes, hidden_dim, num_layers, num_heads, dropout
            num_heads=4,

            # additional TTSDiscriminator inputs: patch_size
            patch_size=patch_size,
        )
    else:
        # initialize an autoencoder-GAN

        # initialize the autoencoder
        ae_dict = torch.load(path_autoencoder, map_location=torch.device('cpu'))
        if ae_dict['configuration']['target'] == 'channels':
            autoencoder = TransformerAutoencoder(input_dim=n_channels,
                                        output_dim=ae_dict['configuration']['channels_out'],
                                        output_dim_2=ae_sequence_length,
                                        target=TransformerAutoencoder.TARGET_CHANNELS,
                                        hidden_dim=ae_dict['configuration']['hidden_dim'],
                                        num_layers=ae_dict['configuration']['num_layers'],
                                        num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
        elif ae_dict['configuration']['target'] == 'time':
            autoencoder = TransformerAutoencoder(input_dim=ae_sequence_length,
                                        output_dim=ae_dict['configuration']['timeseries_out'],
                                        output_dim_2=n_channels,
                                        target=TransformerAutoencoder.TARGET_TIMESERIES,
                                        hidden_dim=ae_dict['configuration']['hidden_dim'],
                                        num_layers=ae_dict['configuration']['num_layers'],
                                        num_heads=ae_dict['configuration']['num_heads'],).to('cpu')
        elif ae_dict['configuration']['target'] == 'full':
            autoencoder = TransformerDoubleAutoencoder(input_dim=n_channels,
                                                output_dim=ae_dict['configuration']['output_dim'],
                                                output_dim_2=ae_dict['configuration']['output_dim_2'],
                                                sequence_length=ae_sequence_length,
                                                hidden_dim=ae_dict['configuration']['hidden_dim'],
                                                num_layers=ae_dict['configuration']['num_layers'],
                                                num_heads=ae_dict['configuration']['num_heads'],
                                                training_level=2).to('cpu')
            
            autoencoder.model_1 = TransformerDoubleAutoencoder(input_dim=n_channels,
                                                output_dim=ae_dict['configuration']['output_dim'],
                                                output_dim_2=ae_dict['configuration']['output_dim_2'],
                                                sequence_length=ae_sequence_length,
                                                hidden_dim=ae_dict['configuration']['hidden_dim'],
                                                num_layers=ae_dict['configuration']['num_layers'],
                                                num_heads=ae_dict['configuration']['num_heads'],
                                                training_level=1).to('cpu')
        else:
            raise ValueError(f"Encode target '{ae_dict['configuration']['target']}' not recognized, options are 'channels', 'time', or 'full'.")
        consume_prefix_in_state_dict_if_present(ae_dict['model'],'module.')
        autoencoder.load_state_dict(ae_dict['model'])
        # freeze the autoencoder
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()

        # if prediction or seq2seq, adjust latent_dim_in to encoded input size
        if input_sequence_length != 0:
            new_input_dim = autoencoder.output_dim if not hasattr(autoencoder, 'output_dim_2') else autoencoder.output_dim*autoencoder.output_dim_2
            latent_dim_in += new_input_dim - autoencoder.input_dim

        # adjust generator output_dim to match the output_dim of the autoencoder
        n_channels = autoencoder.output_dim if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim_2
        sequence_length_generated = autoencoder.output_dim_2 if autoencoder.target in [autoencoder.TARGET_CHANNELS, autoencoder.TARGET_BOTH] else autoencoder.output_dim

        # adjust discriminator input_dim to match the output_dim of the autoencoder
        channel_in_disc = n_channels + n_conditions

        generator = DecoderGenerator(
            generator=gan_architectures[gan_types[gan_type][0]](
                # FFGenerator inputs: latent_dim, output_dim, hidden_dim, num_layers, dropout, activation
                latent_dim=latent_dim_in,
                channels=n_channels,
                seq_len=sequence_length_generated,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                activation=activation,

                # TransformerGenerator inputs: latent_dim, channels, seq_len, hidden_dim, num_layers, num_heads, dropout
                num_heads=4,

                # additional TTSGenerator inputs: patch_size
                patch_size=patch_size,
            ),
            decoder=autoencoder
        )

        discriminator = EncoderDiscriminator(
            discriminator=gan_architectures[gan_types[gan_type][1]](
                # FFDiscriminator inputs: input_dim, hidden_dim, num_layers, dropout
                channels=channel_in_disc,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=0.1,
                seq_len=sequence_length_generated,

                # additional TransformerDiscriminator inputs: num_heads
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