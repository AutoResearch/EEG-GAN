import sys
import traceback
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from autoencoder_training_main import main

if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        'basic': ["data=data/gansMultiCondition_SHORT.csv", "channels_out=2", "time_out=10"],
        'target_time': ["data=data/gansMultiCondition_SHORT.csv", "target=time", "time_out=10"],
        'target_channels': ["data=data/gansMultiCondition_SHORT.csv", "target=channels", "channels_out=1"],
        'target_full': ["data=data/gansMultiCondition_SHORT.csv", "target=full", "time_out=10", "channels_out=1"],
        # 'load_checkpoint': ["data=data/gansMultiCondition.csv", "checkpoint=x"],
    }

    # general parameters
    n_epochs = 1
    batch_size = 32
    kw_channel = "Electrode"
    sample_interval = 1

    for key in configurations.keys():
        try:
            print(f"Running configuration {key}...")
            sys.argv = configurations[key] + [f"n_epochs={n_epochs}", f"batch_size={batch_size}", f"kw_channel={kw_channel}", f"sample_interval={sample_interval}"]
            main()
            print(f"\nConfiguration {key} finished successfully.\n\n")
        # if an error occurs, print key and full error message with traceback and exit
        except:
            print(f"Configuration {key} failed.")
            traceback.print_exc()
            exit(1)
