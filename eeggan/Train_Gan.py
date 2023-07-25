import os
import sys
import warnings
from datetime import datetime
import torch
import torch.multiprocessing as mp

from eeggan.helpers.trainer import Trainer
from eeggan.helpers.get_master import find_free_port
from eeggan.helpers.ddp_training import run, DDPTrainer
from eeggan.nn_architecture.models import TtsDiscriminator, TtsGenerator, TtsGeneratorFiltered
from eeggan.helpers.dataloader import Dataloader
from eeggan.helpers import system_inputs

"""Implementation of the training process of a GAN for the generation of synthetic sequential data.

Instructions to start the training:
  - set the filename of the dataset to load
      - the shape of the dataset should be (n_samples, n_conditions + n_features)
      - the dataset should be a csv file
      - the first columns contain the conditions 
      - the remaining columns contain the time-series data
  - set the configuration parameters (Training configuration; Data configuration; GAN configuration)"""

# TODO: update the training process and the GANs so the distinction between TTS-GAN and other GANs is not necessary

def train_gan(argv = []):
    
    """Main function of the training process."""
    #If run as a function, it receives a dictionary, which will here be converted to match terminal format
    #TODO: Do this in a more standard way?
    if isinstance(argv,dict):
        args = []
        for arg in argv.keys():
            if argv[arg] == True: #If it's a boolean with True
                args.append(str(arg)) #Only include key if it is boolean and true
            elif argv[arg] == False: #If it's a boolean with False
                pass #We do not include the argument if it is turned false
            else: #If it's not a boolean
                args.append(str(arg) + "=" + str(argv[arg])) #Include the key and the value
        argv = args
        
    # sys.argv = ["path_dataset=data/ganAverageERP_len100.csv", "patch_size=20", "conditions=Condition"]
    default_args = system_inputs.parse_arguments(argv, file='Train_Gan.py')
    
    # Shut down if default args = "help"
    if default_args == 'help':
        return []
        
    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    # ----------------------------------------------------------------------------------------------------------------------
    # Configure training parameters and load data
    # ----------------------------------------------------------------------------------------------------------------------

    # Training configuration
    ddp = default_args['ddp']
    ddp_backend = default_args['ddp_backend']
    load_checkpoint = default_args['load_checkpoint']
    path_checkpoint = default_args['path_checkpoint']
    train_gan = default_args['train_gan']
    filter_generator = default_args['filter_generator']

    # trained_embedding = False       # Use an existing embedding
    # use_embedding = False           # Train the embedding in the optimization process

    # Data configuration
    windows_slices = default_args['windows_slices']
    diff_data = False               # Differentiate data
    std_data = False                # Standardize data
    norm_data = True                # Normalize data

    # raise warning if no normalization and standardization is used at the same time
    if std_data and norm_data:
        raise Warning("Standardization and normalization are used at the same time.")

    if (default_args['seq_len_generated'] == -1 or default_args['sequence_length'] == -1) and windows_slices:
        raise ValueError('If window slices are used, the keywords "sequence_length" and "seq_len_generated" must be greater than 0.')

    if load_checkpoint:
        print(f'Resuming training from checkpoint {path_checkpoint}.')

    # Look for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not ddp else torch.device("cpu")
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()

    # GAN configuration
    opt = {
        'n_epochs': default_args['n_epochs'],
        'sequence_length': default_args['sequence_length'],
        'seq_len_generated': default_args['seq_len_generated'],
        'load_checkpoint': default_args['load_checkpoint'],
        'path_checkpoint': default_args['path_checkpoint'],
        'path_dataset': default_args['path_dataset'],
        'batch_size': default_args['batch_size'],
        'learning_rate': default_args['learning_rate'],
        'sample_interval': default_args['sample_interval'],
        'n_conditions': len(default_args['conditions']),
        'patch_size': default_args['patch_size'],
        'kw_timestep': default_args['kw_timestep_dataset'],
        'conditions': default_args['conditions'],
        'lambda_gp': 10,
        'hidden_dim': 128,          # Dimension of hidden layers in discriminator and generator
        'latent_dim': 16,           # Dimension of the latent space
        'critic_iterations': 5,     # number of iterations of the critic per generator iteration for Wasserstein GAN
        'n_lstm': 2,                # number of lstm layers for lstm GAN
        'world_size': world_size,   # number of processes for distributed training
    }

    # Load dataset as tensor
    dataloader = Dataloader(default_args['path_dataset'],
                            kw_timestep=default_args['kw_timestep_dataset'],
                            col_label=default_args['conditions'],
                            norm_data=norm_data,
                            std_data=std_data,
                            diff_data=diff_data)
    dataset = dataloader.get_data(sequence_length=default_args['sequence_length'],
                                  windows_slices=default_args['windows_slices'], stride=5,
                                  pre_pad=default_args['sequence_length']-default_args['seq_len_generated'])
    opt['sequence_length'] = dataset.shape[1] - dataloader.labels.shape[1]
    opt['n_samples'] = dataset.shape[0]

    # keep randomly 30% of the data
    # dataset = dataset[np.random.randint(0, dataset.shape[0], int(dataset.shape[0]*0.3))]

    if opt['sequence_length'] % opt['patch_size'] != 0:
        warnings.warn(f"Sequence length ({opt['sequence_length']}) must be a multiple of patch size ({default_args['patch_size']}).\n"
                      f"The sequence length is padded with zeros to fit the condition.")
        padding = 0
        while opt['sequence_length'] % default_args['patch_size'] != 0:
            padding += 1
        opt['sequence_length'] += padding
        dataset = torch.cat((dataset, torch.zeros(dataset.shape[0], padding)), dim=1)

    if opt['seq_len_generated'] == -1:
        opt['seq_len_generated'] = opt['sequence_length']

    # Embedding network to reduce the dimension of time-series data
    # not tested yet
    # if use_embedding:
    #     # Use pretrained embedding
    #     if trained_embedding:
    #         # load encoder
    #         encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         encoder_weights = torch.load(r'trained_models\embedding_encoder.pt')
    #         encoder.load_state_dict(encoder_weights)
    #         # load decoder
    #         decoder = Decoder(output_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         decoder_weights = torch.load(r'trained_models\\embedding_decoder.pt')
    #         decoder.load_state_dict(decoder_weights)
    #         print('Loaded pretrained embedding.')
    #     else:
    #         # train embedding
    #         print('Training embedding...')
    #         encoder = Encoder(input_size=2, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'])
    #         decoder = Decoder(signals=1, conditions=1, hidden_size=opt['hidden_dim'], embedding_dim=opt['latent_dim'],
    #                           seq_len=dataset.shape[1]-opt['n_conditions'])
    #         embedding_trainer = EmbeddingNetTrainer(encoder, decoder, opt)
    #         encoder, decoder, emb_samples, losses = embedding_trainer.train(dataset)
    #         print('Finished training embedding.')
    #         plt.plot(losses)
    #         plt.show()
    #
    #         # save embedding
    #         # pickle emb_samples
    #         # with open('emb_samples.pkl', 'wb') as f:
    #         #     pickle.dump(emb_samples, f)
    #         df = pd.DataFrame(emb_samples, columns=None, index=None).T
    #         torch.save(encoder.state_dict(), 'trained_models/encoder_' + timestamp + '.pt')
    #         torch.save(decoder.state_dict(), 'trained_models/decoder_' + timestamp + '.pt')

    # Initialize generator, discriminator and trainer

    if not filter_generator:
        generator = TtsGenerator(seq_length=opt['seq_len_generated'],
                                 latent_dim=opt['latent_dim'] + opt['n_conditions'] + opt['sequence_length'] - opt['seq_len_generated'],
                                 patch_size=opt['patch_size'],
                                 channels=1)  # TODO: Channel recovery: set channels to number of channels in dataset
    else:
        generator = TtsGeneratorFiltered(seq_length=opt['seq_len_generated'],
                                         latent_dim=opt['latent_dim']+opt['n_conditions']+opt['sequence_length']-opt['seq_len_generated'],
                                         patch_size=opt['patch_size'])
    discriminator = TtsDiscriminator(seq_length=opt['sequence_length'], patch_size=opt['patch_size'], in_channels=1+opt['n_conditions'])  # TODO: Channel recovery: set in_channels to (number of channels)*2 in dataset
    print("Generator and discriminator initialized.")

    # ----------------------------------------------------------------------------------------------------------------------
    # Start training process
    # ----------------------------------------------------------------------------------------------------------------------

    if train_gan:
        # GAN-Training
        print('\n-----------------------------------------')
        print("Training GAN...")
        print('-----------------------------------------\n')
        if ddp:
            trainer = DDPTrainer(generator, discriminator, opt)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['path_checkpoint'])
            mp.spawn(run, args=(world_size, find_free_port(), ddp_backend, trainer, opt),
                     nprocs=world_size, join=True)
        else:
            trainer = Trainer(generator, discriminator, opt)
            if default_args['load_checkpoint']:
                trainer.load_checkpoint(default_args['path_checkpoint'])
            gen_samples = trainer.training(dataset)

            # save final models, optimizer states, generated samples, losses and configuration as final result
            if default_args['path_checkpoint']:
                trainer.save_checkpoint(path_checkpoint=default_args['path_checkpoint'], generated_samples=gen_samples)
            else:
                path = 'trained_models'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'gan_{trainer.epochs}ep_' + timestamp + '.pt'
                trainer.save_checkpoint(path_checkpoint=os.path.join(path, filename), generated_samples=gen_samples)

        print("GAN training finished.")
        print("Generated samples saved to file.")
        print("Model states saved to file.")
    else:
        print("GAN not trained.")

if __name__ == '__main__':
    train_gan(sys.argv)
    
