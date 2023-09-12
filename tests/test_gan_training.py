import sys
import traceback
from gan_training_main import main

if __name__ == '__main__':
    configurations = {
        # configurations for normal GAN
        # 'basic': ["path_dataset=../data/gansMultiCondition_SHORT.csv"],
        # '1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "conditions=Condition"],
        # '2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "conditions=Trial,Condition"],
        # '2channel': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "channel_label=Electrode"],
        # '2channel_1condition': ["sample_interval=1", "path_dataset=../data/gansMultiCondition_SHORT.csv", "channel_label=Electrode", "conditions=Condition"],
        # '2channel_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "channel_label=Electrode", "conditions=Trial,Condition"],
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
        # 'autoencoder_basic': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "channel_label=Electrode"],
        # 'autoencoder_1condition': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "load_checkpoint", "path_checkpoint=../trained_models/checkpoint.pt", "path_autoencoder=../trained_ae/ae_gansMultiCondition_SHORT.pt", "channel_label=Electrode", "conditions=Condition"],
        # 'autoencoder_2conditions': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "path_autoencoder=../trained_ae/ae_gansMultiCondition_SHORT.pt", "channel_label=Electrode", "conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_2conditions_channels': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "path_autoencoder=../trained_ae/ae_gansMultiCondition_SHORT_channels.pt", "channel_label=Electrode", "conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        'autoencoder_2conditions_time': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "path_autoencoder=../trained_ae/ae_gansMultiCondition_SHORT_time.pt", "channel_label=Electrode", "conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        'autoencoder_2conditions_full': ["path_dataset=../data/gansMultiCondition_SHORT.csv", "path_autoencoder=../trained_ae/ae_gansMultiCondition_SHORT_full.pt", "channel_label=Electrode", "conditions=Trial,Condition", "hidden_dim=64", "activation=leakyrelu", "num_layers=1",],
        # 'autoencoder_prediction': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=70", "channel_label=Electrode"],
        # 'autoencoder_prediction_1condition': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=70", "channel_label=Electrode", "conditions=Condition"],
        # 'autoencoder_prediction_2conditions': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=70", "channel_label=Electrode", "conditions=Trial,Condition"],
        # 'autoencoder_seq2seq': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=-1", "channel_label=Electrode"],
        # 'autoencoder_seq2seq_1condition': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Condition"],
        # 'autoencoder_seq2seq_2conditions': ["path_dataset=../data/ganTrialElectrodeERP_p50_e8_len100_SHORT.csv", "path_autoencoder=../trained_ae/transformer2D_ae.pt", "input_sequence_length=-1", "channel_label=Electrode", "conditions=Trial,Condition"],
    }
    
    # general parameters
    n_epochs = 1
    batch_size = 32
    
    key = None
    try:
        for key in configurations.keys():
            print(f"Running configuration {key}...")
            sys.argv = configurations[key] + [f"n_epochs={n_epochs}", f"batch_size={batch_size}"]
            generator, discriminator, opt, gen_samples = main()
            print(f"\nConfiguration {key} finished successfully.\n\n")
    # if an error occurs, print key and full error message with traceback and exit
    except:
        print(f"Configuration {key} failed.")
        traceback.print_exc()
        exit(1)
