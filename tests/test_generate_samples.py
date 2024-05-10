import os
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from generate_samples_main import main

if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        'basic': ["model=trained_models/gan_basic.pt"],
        '1condition': ["model=trained_models/gan_1cond.pt", "conditions=0"],
        '2conditions': ["model=trained_models/gan_2cond.pt", "conditions=0,1"],
        '2channel': ["model=trained_models/gan_2ch.pt", "save_name=generated_samples/gan_2ch.csv"],
        '2channel_1condition': ["model=trained_models/gan_2ch_1cond.pt", "conditions=0"],
        '2channel_2conditions': ["model=trained_models/gan_2ch_2cond.pt", "conditions=0,1"],
        
        # configurations for autoencoder GAN
        'ae_basic': ["model=trained_models/gan_ae.pt"],
        'ae_1condition': ["model=trained_models/gan_ae_1cond.pt", "conditions=0"],
        'ae_2conditions': ["model=trained_models/gan_ae_2cond.pt", "conditions=0,1"],
        'ae_2channel': ["model=trained_models/gan_ae_2ch.pt"],
        'ae_2channel_1condition': ["model=trained_models/gan_ae_2ch_1cond.pt", "conditions=0"],
        'ae_2channel_2conditions': ["model=trained_models/gan_ae_2ch_2cond.pt", "conditions=0,1"],
    }

    key = None
    try:
        for key in configurations.keys():
            print(f"Running configuration {key}...")
            sys.argv = configurations[key]
            main()
            print(f"\nConfiguration {key} finished successfully.\n\n")
    # if an error occurs, print key and full error message with traceback and exit
    except:
        print(f"Configuration {key} failed.")
        traceback.print_exc()
        exit(1)
