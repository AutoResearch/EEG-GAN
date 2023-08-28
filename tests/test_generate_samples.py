import sys
import traceback
from generate_samples_main import main

if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        # 'basic': ["file=..\\trained_models\\gan_1ep_basic.pt", "path_samples=..\\generated_samples\\gan_1ep_basic.csv"],
        # '1condition': ["file=..\\trained_models\\gan_1ep_1cond.pt", "path_samples=..\\generated_samples\\gan_1ep_1cond.csv", "conditions=0"],
        # '2conditions': ["file=..\\trained_models\\gan_1ep_2cond.pt", "path_samples=..\\generated_samples\\gan_1ep_2cond.csv", "conditions=0,1"],
        # '2channel': ["file=..\\trained_models\\gan_1ep_2chan.pt", "path_samples=..\\generated_samples\\gan_1ep_2chan.csv"],
        # '2channel_1condition': ["file=..\\trained_models\\gan_1ep_2chan_1cond.pt", "path_samples=..\\generated_samples\\gan_1ep_2chan_1cond.csv", "conditions=0"],

        '2channel_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "channel_label=Electrode", "conditions=Trial,Condition"],
        # 'prediction': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70"],
        # 'prediction_1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "conditions=Condition"],
        # 'prediction_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "conditions=Trial,Condition"],
        # 'prediction_2channel': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "channel_label=Electrode"],
        # 'prediction_2channel_1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "channel_label=Electrode", "conditions=Condition"],
        # 'prediction_2channel_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "channel_label=Electrode", "conditions=Trial,Condition"],
        # 'seq2seq': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1"],
        # 'seq2seq_1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "conditions=Condition"],
        # 'seq2seq_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "conditions=Trial,Condition"],
        # 'seq2seq_2channel': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "channel_label=Electrode"],
        # 'seq2seq_2channel_1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Condition"],
        # 'seq2seq_2channel_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Trial,Condition"],

        # configurations for autoencoder GAN
        # 'autoencoder_basic': ["file=..\\trained_models\\gan_1ep_ae.pt", "path_samples=..\\generated_samples\\gan_1ep_ae.csv"],
        # 'autoencoder_1condition': ["file=..\\trained_models\\gan_1ep_ae_1cond.pt", "path_samples=..\\generated_samples\\gan_1ep_ae_1cond.csv", "conditions=0"],
        # 'autoencoder_2conditions': ["file=..\\trained_models\\gan_1ep_ae_2cond.pt", "path_samples=..\\generated_samples\\gan_1ep_ae_2cond.csv", "conditions=0,1"],

        # 'autoencoder_prediction': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=70", "channel_label=Electrode"],
        # 'autoencoder_prediction_1condition': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=70", "channel_label=Electrode", "conditions=Condition"],
        # 'autoencoder_prediction_2conditions': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=70", "channel_label=Electrode", "conditions=Trial,Condition"],
        # 'autoencoder_seq2seq': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=-1", "channel_label=Electrode"],
        # 'autoencoder_seq2seq_1condition': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Condition"],
        # 'autoencoder_seq2seq_2conditions': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "gan_type=autoencoder", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Trial,Condition"],
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
