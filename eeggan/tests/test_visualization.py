import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from visualize_main import main

if __name__ == '__main__':
    configurations = {
        # configuration for real data
        'dataset': ["data=data/testMultiConditionMultiChannel.csv"],
        'dataset_2condition': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2"],
        'dataset_2condition_2channels': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel"],
        'dataset_2condition_channelplot': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "channel_plots"],
        'dataset_2condition_2channels_channelindex': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel", "channel_index=0"],
        'dataset_2condition_2channels_channelplot': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel", "channel_plots"],        
        'dataset_avg': ["data=data/testMultiConditionMultiChannel.csv", "average"],
        'dataset_2condition_avg': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "average"],
        'dataset_1condition_2channels_avg': ["data=data/testMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel", "average"],
        'dataset_pca': ["data=data/testMultiConditionMultiChannel.csv", "pca", "comp_data=data/testCompMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel"],
        'dataset_tsne': ["data=data/testMultiConditionMultiChannel.csv", "tsne", "comp_data=data/testCompMultiConditionMultiChannel.csv", "kw_conditions=Cond1,Cond2", "kw_channel=Channel"],
        
        # configurations for synthetic data
        'synt': ["data=generated_samples/gan_basic.csv"],
        'synt_ae': ["data=generated_samples/gan_ae_2ch.csv", "kw_channel=Electrode"],
        'synt_spectogram': ["data=generated_samples/gan_ae_2ch.csv", "kw_channel=Electrode", "spectogram"],
        'synt_fft': ["data=generated_samples/gan_ae_2ch.csv", "kw_channel=Electrode", "fft"],
        
        # configurations for normal GAN
        'basic': ["model=trained_models/gan_basic.pt"],
        'basic_2condition': ["model=trained_models/gan_2cond.pt"],
        'basic_2condition_2channels': ["model=trained_models/gan_2ch_2cond.pt"],
        'basic_2condition_2_channels_channelplot': ["model=trained_models/gan_2ch_2cond.pt", "channel_plots"],
        
        # configurations for autoencoder GAN
        'gan_ae_basic': ["model=trained_models/gan_ae.pt"],
        
        # configuration for autoencoder
        'ae_basic': ["model=trained_ae/ae_target_full.pt"],
        'ae_pca': ["model=trained_ae/ae_target_full.pt", "pca"],
    }

    n_samples = 4
    
    key = None
    try:
        for key in configurations.keys():
            print(f"Running configuration {key}...")
            sys.argv = configurations[key] + [f"n_samples={n_samples}"]
            main()
            print(f"\nConfiguration {key} finished successfully.\n\n")
    # if an error occurs, print key and full error message with traceback and exit
    except:
        print(f"Configuration {key} failed.")
        traceback.print_exc()
        exit(1)
