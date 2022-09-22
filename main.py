import os
from datetime import datetime
import pandas as pd
import torch
from matplotlib import pyplot as plt

from cwgan import Trainer
from models import TtsDiscriminator, TtsGenerator
from dataloader import Dataloader
from EmbeddingNet import Encoder, Decoder, Trainer as EmbeddingNetTrainer

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""

# TODO: update the training process and the GANs so the distinction between TTS-GAN and other GANs is not necessary

# ----------------------------------------------------------------------------------------------------------------------
# Configure training parameters and load data
# ----------------------------------------------------------------------------------------------------------------------

# Training configuration
trained_gan = False             # Use an existing GAN/Checkpoints of previous training
train_gan = True                # Train the GAN in the optimization process
trained_embedding = False       # Use an existing embedding
use_embedding = False           # Train the embedding in the optimization process

# Data configuration
diff_data = False               # Differentiate data
std_data = False                # Standardize data
norm_data = True                # Normalize data

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

print("Dataset configuration:\n\tdifferentiation: {}\n\tstandardization: {}\n\tnormalization: {}"
      .format(diff_data, std_data, norm_data))
# raise warning if no normalization and standardization is used at the same time
if std_data and norm_data:
    raise Warning("Standardization and normalization are used at the same time.")

# Look for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAN configuration
opt = {
        'n_epochs': 2,              # number of training epochs of batch training
        'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
        'batch_size': 6,            # batch size for batch training
        'learning_rate': 1e-4,      # learning rate of the generator and discriminator
        'latent_dim': 16,           # Dimension of the latent space
        'sample_interval': 1,       # interval between recorded samples
        'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_conditions': 1,          # number of conditions for conditional GAN
        'n_lstm': 2,                # number of lstm layers for lstm GAN
        'patch_size': 15            # Patch size for the transformer GAN (tts-gan)
    }

# Load dataset as tensor
path = r'\data\ganAverageERP_mini.csv'
dataloader = Dataloader(path, diff_data, std_data, norm_data)
dataset = dataloader.get_data()
# make sequence length of dataset dividable by 15 by padding with zeros
while (dataset.shape[1] - opt['n_conditions']) % opt['patch_size'] != 0:
    padding = torch.zeros(dataset.shape[0], 1)
    dataset = torch.cat((dataset, padding), dim=1)

# Embedding network to reduce the dimension of time-series data
# not tested yet
if use_embedding:
    # Use pretrained embedding
    if trained_embedding:
        # load encoder
        encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
        encoder_weights = torch.load(r'trained_models\embedding_encoder.pt')
        encoder.load_state_dict(encoder_weights)
        # load decoder
        decoder = Decoder(output_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
        decoder_weights = torch.load(r'trained_models\\embedding_decoder.pt')
        decoder.load_state_dict(decoder_weights)
        print('Loaded pretrained embedding.')
    else:
        # train embedding
        print('Training embedding...')
        encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
        decoder = Decoder(signals=1, conditions=1, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'],
                          seq_len=dataset.shape[1]-opt['n_conditions'])
        embedding_trainer = EmbeddingNetTrainer(encoder, decoder, opt)
        encoder, decoder, emb_samples, losses = embedding_trainer.train(dataset)
        print('Finished training embedding.')
        plt.plot(losses)
        plt.show()

        # save embedding
        # pickle emb_samples
        # with open('emb_samples.pkl', 'wb') as f:
        #     pickle.dump(emb_samples, f)
        df = pd.DataFrame(emb_samples, columns=None, index=None).T
        torch.save(encoder.state_dict(), 'trained_models/encoder_' + timestamp + '.pt')
        torch.save(decoder.state_dict(), 'trained_models/decoder_' + timestamp + '.pt')

# Initialize generator, discriminator and trainer
state_dict = None
optG = None
optD = None
generator = TtsGenerator(seq_length=dataset.shape[1]-opt['n_conditions'],
                         latent_dim=opt['latent_dim']+opt['n_conditions']).to(device)
discriminator = TtsDiscriminator(seq_length=dataset.shape[1]-opt['n_conditions']).to(device)
trainer = Trainer(generator, discriminator, opt, optimizer_generator=optG, optimizer_discriminator=optD)
print("Generator and discriminator initialized.")

if trained_gan:
    # Use pretrained generator and discriminator

    # check if checkpoint-file exists
    if os.path.isfile(r'trained_models\checkpoint.pth'):
        # load state_dicts
        state_dict = torch.load(r'trained_models\checkpoint.pth')
        generator.load_state_dict(state_dict['generator'])
        discriminator.load_state_dict(state_dict['discriminator'])
        optG = state_dict['optimizer_generator']
        optD = state_dict['optimizer_discriminator']
        print("Using pretrained GAN.")
    else:
        Warning("No checkpoint-file found. If you do not wish to continue training, set trained_gan to False.")

# ----------------------------------------------------------------------------------------------------------------------
# Start training process
# ----------------------------------------------------------------------------------------------------------------------

if train_gan:
    # GAN-Training
    print("Training GAN...")
    generator, discriminator, gen_samples = trainer.batch_train(dataset)

    # Apply data re-transformation
    for i in range(len(gen_samples)):
        if len(gen_samples[i].shape) > 2:
            gen_samples[i] = gen_samples[i].reshape(gen_samples[i].shape[-1],)
        if diff_data:
            gen_samples[i] = dataloader.inverse_diff(gen_samples[i], dim=1)
        if std_data:
            gen_samples[i] = dataloader.inverse_std(gen_samples[i])
        if norm_data:
            gen_samples[i] = dataloader.inverse_norm(gen_samples[i])

    # Append generated samples to dataframe
    gen_samples = pd.DataFrame(gen_samples)

    # Save trained_models for later use
    gen_samples.to_csv(r'generated_samples\sample_' + timestamp + '.csv', index_label=False)
    state_dict = {
        'generator': trainer.generator.state_dict(),
        'discriminator': trainer.discriminator.state_dict(),
        'optimizer_generator': trainer.generator_optimizer.state_dict(),
        'optimizer_discriminator': trainer.discriminator_optimizer.state_dict(),
    }
    torch.save(state_dict, r'trained_models\state_dict_' + timestamp + '.pt')
    
    print("GAN training finished.")
    print("Generated samples saved to file.")
    print("Model states saved to file.")
else:
    print("GAN not trained.")
    