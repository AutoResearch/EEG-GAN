import sys
import traceback
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from gan_training_main import main


if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        'basic': ["data=data/gansMultiCondition_SHORT.csv"],
        '1condition': ["data=data/gansMultiCondition_SHORT.csv", "kw_conditions=Condition", "kw_channel=Electrode"],
        'load_checkpoint': ["data=data/gansMultiCondition_SHORT.csv", "checkpoint=x", "kw_conditions=Condition", "kw_channel=Electrode"]
        # '2conditions': ["data=data/gansMultiCondition_SHORT.csv", "kw_conditions=Trial,Condition"],
        # '2channels': ["data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode"],
        # '2channels_1condition': ["sample_interval=1", "data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode", "kw_conditions=Condition"],
        # '2channels_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "kw_channel=Electrode", "kw_conditions=Trial,Condition"],
        # 'prediction': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=70"],
        # 'prediction_1condition': ["data=data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "kw_conditions=Condition"],
        # 'prediction_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "kw_conditions=Trial,Condition"],
        # 'prediction_2channels': ["data=data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "kw_channel=Electrode"],
        # 'prediction_2channels_1condition': ["data=data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "kw_channel=Electrode", "kw_conditions=Condition"],
        # 'prediction_2channels_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "patch_size=20", "input_sequence_length=70", "kw_channel=Electrode", "kw_conditions=Trial,Condition"],
        # 'seq2seq': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1"],
        # 'seq2seq_1condition': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "kw_conditions=Condition"],
        # 'seq2seq_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "kw_conditions=Trial,Condition"],
        # 'seq2seq_2channels': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "kw_channel=Electrode"],
        # 'seq2seq_2channels_1condition': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "kw_channel=Electrode", "kw_conditions=Condition"],
        # 'seq2seq_2channels_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "input_sequence_length=-1", "kw_channel=Electrode", "kw_conditions=Trial,Condition"],

        # configurations for autoencoder GAN
        # 'autoencoder_basic': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "kw_channel=Electrode"],
        # 'autoencoder_1condition': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "kw_channel=Electrode", "kw_conditions=Condition"],
        # 'autoencoder_2conditions': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_2conditions_channels': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_channels.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_2conditions_time': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_time.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_2conditions_full': ["data=data/gansMultiCondition_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "kw_channel=Electrode", "kw_conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_prediction': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=70", "kw_channel=Electrode"],
        # 'autoencoder_prediction_1condition': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=70", "kw_channel=Electrode", "kw_conditions=Condition"],
        # 'autoencoder_prediction_2conditions': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=70", "kw_channel=Electrode", "kw_conditions=Trial,Condition"],
        # 'autoencoder_seq2seq': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=-1", "kw_channel=Electrode"],
        # 'autoencoder_seq2seq_1condition': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=-1", "kw_channel=Electrode", "kw_conditions=Condition"],
        # 'autoencoder_seq2seq_2conditions': ["data=data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=trained_ae/ae_gansMultiCondition_SHORT_full.pt", "input_sequence_length=-1", "kw_channel=Electrode", "kw_conditions=Trial,Condition"],
    }
    
    # general parameters
    n_epochs = 1
    batch_size = 32
    gan_type = ['tts','ff','tr']
    patch_size = 10

    for gan in gan_type:
        key = None
        try:
            for key in configurations.keys():
                print(f"Running configuration {key}...")
                sys.argv = configurations[key] + [f"n_epochs={n_epochs}", f"batch_size={batch_size}", f"type={gan}", f"patch_size={patch_size}"]
                generator, discriminator, opt, gen_samples = main()
                print(f"\nConfiguration {key} finished successfully.\n\n")
        # if an error occurs, print key and full error message with traceback and exit
        except:
            print(f"Configuration {key} failed.")
            traceback.print_exc()
            exit(1)
