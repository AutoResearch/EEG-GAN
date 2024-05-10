import sys
import traceback
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from gan_training_main import main


if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        'basic': ["data=data/gansMultiCondition_SHORT.csv", "save_name=gan_basic.pt"],
        '1condition': ["data=data/gansMultiCondition_SHORT.csv", "kw_conditions=Condition", "save_name=gan_1cond.pt"],
        'load_checkpoint': ["data=data/gansMultiCondition_SHORT.csv", "checkpoint=x", "kw_conditions=Condition"],
        '2conditions': ["data=data/gansMultiCondition_SHORT.csv", "kw_conditions=Trial,Condition", "save_name=gan_2cond.pt"],
        '2channels': ["data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode", "save_name=gan_2ch.pt"],
        '2channels_1condition': ["sample_interval=1", "data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode", "kw_conditions=Condition", "save_name=gan_2ch_1cond.pt"],
        '2channels_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "save_name=gan_2ch_2cond.pt"],
        
        # configurations for autoencoder GAN
        'autoencoder_basic': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "save_name=gan_ae.pt"],
        'autoencoder_1condition': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "kw_conditions=Condition", "save_name=gan_ae_1cond.pt"],
        'autoencoder_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "save_name=gan_ae_2cond.pt"],
        'autoencoder_2channels': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "save_name=gan_ae_2ch.pt"],
        'autoencoder_2channels_1conditions': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "kw_conditions=Condition", "save_name=gan_ae_2ch_1cond.pt"],
        'autoencoder_2channels_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "save_name=gan_ae_2ch_2cond.pt"],
        
        # 'autoencoder_2conditions_channels': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT_channels.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "num_layers=1",],
        # 'autoencoder_2conditions_time': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT_time.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "num_layers=1",],
        # 'autoencoder_2conditions_full': ["data=data/gansMultiCondition_SHORT.csv", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "num_layers=1",],
        # 'load_checkpoint': ["data=data/gansMultiCondition_SHORT.csv", "checkpoint=x", "autoencoder=trained_ae/ae_gansMultiCondition_SHORT.pt", "kw_conditions=Condition", "kw_channel=Electrode"],
    }
    
    # general parameters
    n_epochs = 1
    batch_size = 32
    patch_size = 10

    key = None
    try:
        for key in configurations.keys():
            print(f"Running configuration {key}...")
            sys.argv = configurations[key] + [f"n_epochs={n_epochs}", f"batch_size={batch_size}", f"patch_size={patch_size}"]
            generator, discriminator, opt, gen_samples = main()
            print(f"\nConfiguration {key} finished successfully.\n\n")
    # if an error occurs, print key and full error message with traceback and exit
    except:
        print(f"Configuration {key} failed.")
        traceback.print_exc()
        exit(1)
