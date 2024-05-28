"""This file shows which inputs can be given to gan_training_main.py from the command line."""
import os
import sys
from typing import List, Union


class Helper:
    def __init__(self, kw_dict):
        self.kw_dict = kw_dict

        if self.kw_dict is not None:
            # Check if default values are of the correct type
            for key, value in kw_dict.items():
                if value[2] is not None and type(value[2]) != value[0]:
                    raise TypeError(
                        f'Default value of {key} is not of given type {value[0]}. Please correct the default value.')

    def print_table(self):
        # Define what to print in table
        if self.kw_dict is not None:
            values = list(self.kw_dict.keys())
            types = [self.kw_dict[key][0] for key in values]
            descriptions = [self.kw_dict[key][1] for key in values]
            defaults = [self.kw_dict[key][2] for key in values]

            # Get longest string length of each column
            max_len = [0, 0, 0, 0]
            for i in range(len(self.kw_dict)):
                max_len[0] = max(max_len[0], len(values[i]))
                max_len[1] = max(max_len[1], len(str(types[i])))
                max_len[2] = max(max_len[2], len(descriptions[i]))
                max_len[3] = max(max_len[3], len(str(defaults[i])))

            # Print table
            self.start_line()
            print(f"INPUT HELP - These are the inputs that can be given from the command line")
            self.end_line()
            # print header of table
            print(
                f'{"Input":<{max_len[0]}} | {"Type":<{max_len[1]}} | {"Description":<{max_len[2]}} | {"Default value":<{max_len[3]}}')
            self.line()
            # Print values of table
            for i in range(len(values)):
                # define spacing for each column
                spacing = [' ' * (max_len[0] - len(values[i])), ' ' * (max_len[1] - len(str(types[i]))),
                           ' ' * (max_len[2] - len(descriptions[i])), ' ' * (max_len[3] - len(str(defaults[i])))]
                # print row
                print(
                    f'{values[i]}{spacing[0]} | {str(types[i])}{spacing[1]} | {descriptions[i]}{spacing[2]} | {defaults[i]}{spacing[3]}')

            # print end of table
            self.line()
        else:
            print("No inputs found.")

    def print_help(self):
        self.start_line()
        print("QUICK HELP - These are the special features:")
        self.line()
        print('General information: '
              '\n'
              '\nBoolean arguments are given as a single keyword:'
              '\n\tSet boolean keyword "test_keyword" to True\t->\tpython file.py test_keyword'
              '\nCommand line arguments are given as a keyword followed by an equal sign and the value:'
              '\n\tSet command line argument "test_keyword" to "test_value"\t->\tpython file.py test_keyword=test_value'
              '\n\tWhitespaces are not allowed between a keyword and its value.'
              '\nSome keywords can be given list-like: '
              '\n\ttest_keyword=test_value1,test_value2'
              '\n\tThese keywords are marked with ** in the table.')
        self.line()

    def start_line(self):
        print('\n')
        self.line()

    def end_line(self):
        self.line()
        print('\n')

    @staticmethod
    def line():
            print('----------------------------------------------------------------------------------------------------------------------------------------------------')


class HelperMain(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        """Print help message for gan_training_main.py regarding special features."""
        super().print_help()
        print(
            '1.\tThe training works with two levels of checkpoint files:'
            '\n\t1.1 During the training:'
            '\n\t\tCheckpoints are saved every "sample_interval" batches as either "checkpoint_01.pt"'
            '\n\t\tor "checkpoint_02.pt". These checkpoints are considered as low-level checkpoints since they are only '
            '\n\t\tnecessary in the case of training interruption. Hereby, they can be used to continue the training from '
            '\n\t\tthe most recent sample. To continue training, the most recent checkpoint file must be renamed to '
            '\n\t\t"checkpoint.pt".'
            '\n\t\tFurther, these low-level checkpoints carry the generated samples for inference purposes.'
            '\n\t1.2 After finishing the training:'
            '\n\t\tA high-level checkpoint is saved as "checkpoint.pt", which is used to '
            '\n\t\tcontinue training in another session. This high-level checkpoint does not carry the generated samples.'
            '\n\t\tTo continue training from this checkpoint file no further adjustments are necessary. '
            '\n\t\tSimply give the keyword "load_checkpoint" when calling the training process.'
            '\n\t\tThe low-level checkpoints are deleted after creating the high-level checkpoint.'
            '\n\t1.3 For inference purposes:'
            '\n\t\tAnother dictionary is saved as "gan_{n_epochs}ep_{timestamp}.pt".'
            '\n\t\tThis file contains everything the checkpoint file contains, plus the generated samples.')
        print(
            '2.\tUse "ddp" to activate distributed training. '
            '\n\tOnly if multiple GPUs are available for one node.'
            '\n\tAll available GPUs are used for training.'
            '\n\tEach GPU trains on the whole dataset. '
            '\n\tHence, the number of training epochs is multiplied by the number of GPUs')
        print(
            '3.\tIf you want to load a pre-trained GAN, you can use the following command:'
            '\n\tpython gan_training_main.py load_checkpoint; The default file is "trained_models/checkpoint.pt"'
            '\n\tIf you want to use another file, you can use the following command:'
            '\n\t\tpython gan_training_main.py load_checkpoint path_checkpoint="path/to/file.pt"')
        print(
            '4.\tIf you want to use a different dataset, you can use the following command:'
            '\n\tpython gan_training_main.py data="path/to/file.csv"'
            '\n\tThe default dataset is "data/gansEEGTrainingData.csv"')
        print(
            '6.\tThe keyword "input_sequence_length" describes the length of a sequence taken as input for the generator.'
            '\n\t6.1 The "input_sequence_length" must be smaller than the total sequence length.'
            '\n\t6.2 The generator works in the following manner:'
            '\n\t\tThe generator gets a sequence of length "input_sequence_length" as a condition (input).'
            '\n\t\tThe generator generates a sequence of length "sequence_length"-"input_sequence_length" as output which is used as the '
            '\n\t\tsubsequent part of the input sequence.'
            '\n\t6.3 If "input_sequence_length" == 0:'
            '\n\t\tThe generator does not get any input sequence but generates an arbitrary sequence of length "sequence_length".'
            '\n\t\tArbitrary means hereby that the generator does not get any conditions on previous data points.')
        self.start_line()
        self.end_line()


class HelperVAE(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        """Print help message for vae_training_main.py regarding special features."""
        super().print_help()
        print(
            '1.\tThe training works with two levels of checkpoint files:'
            '\n\t1.1 During the training:'
            '\n\t\tCheckpoints are saved every "sample_interval" batches as either "checkpoint_01.pt"'
            '\n\t\tor "checkpoint_02.pt". These checkpoints are considered as low-level checkpoints since they are only '
            '\n\t\tnecessary in the case of training interruption. Hereby, they can be used to continue the training from '
            '\n\t\tthe most recent sample. To continue training, the most recent checkpoint file must be renamed to '
            '\n\t\t"checkpoint.pt".'
            '\n\t\tFurther, these low-level checkpoints carry the generated samples for inference purposes.'
            '\n\t1.2 After finishing the training:'
            '\n\t\tA high-level checkpoint is saved as "checkpoint.pt", which is used to '
            '\n\t\tcontinue training in another session. This high-level checkpoint does not carry the generated samples.'
            '\n\t\tTo continue training from this checkpoint file no further adjustments are necessary. '
            '\n\t\tSimply give the keyword "load_checkpoint" when calling the training process.'
            '\n\t\tThe low-level checkpoints are deleted after creating the high-level checkpoint.'
            '\n\t1.3 For inference purposes:'
            '\n\t\tAnother dictionary is saved as "vae_{n_epochs}ep_{timestamp}.pt".'
            '\n\t\tThis file contains everything the checkpoint file contains, plus the generated samples.')
        print(
            '2.\tUse "ddp" to activate distributed training. '
            '\n\tOnly if multiple GPUs are available for one node.'
            '\n\tAll available GPUs are used for training.'
            '\n\tEach GPU trains on the whole dataset. '
            '\n\tHence, the number of training epochs is multiplied by the number of GPUs')
        print(
            '3.\tIf you want to load a pre-trained VAE, you can use the following command:'
            '\n\tpython gan_training_main.py load_checkpoint; The default file is "trained_models/checkpoint.pt"'
            '\n\tIf you want to use another file, you can use the following command:'
            '\n\t\tpython vae_training_main.py load_checkpoint path_checkpoint="path/to/file.pt"')
        print(
            '4.\tIf you want to use a different dataset, you can use the following command:'
            '\n\tpython gan_training_main.py path_dataset="path/to/file.csv"'
            '\n\tThe default dataset is "data/gansEEGTrainingData.csv"')
        self.start_line()
        self.end_line()

class HelperAutoencoder(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tThe target parameter determines whether you will encode the channels, timeseries, or both (named full):'
              '\n\tIf target = channels, then the channels_out parameter will be used'
              '\n\tIf target = timeseries, then the timeseries_out parameter will be used'
              '\n\tif target = full, then both the channels_out and timeseries_out parameters will be used')
        print('2.\tThe channels_out and timeseries_out parameters indicate the corresponding dimension size output of the encoder'
              '\n\t\tFor example, if we havea a 100x30 (timeseries x channel) sample and use timeseries_out=10 & channels_out=4'
              '\n\t\twith target=full, our encoder will result in an encoded 10x4 sample')
        print('3.\t"load_checkpoint" can be used to load a previously trained autoencoder model and continue training on it.'
              '\n\t3.1 If you are loading a previously trained model, it will inherit the following model parameters:'
              '\n\t\ttarget, channels_out, timeseries_out. The remainder of the parameters will be used as normal.'
              '\n\t3.2 If you do not specify "path_checkpoint" the default path is "trained_ae/checkpoint.pt"')


class HelperVisualize(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tEither the keyword "model" or "data" must be given.'
              '\n\t1.1 If the keyword "model" is given'
              '\n\t\t"model" must point to a pt-file.'
              '\n\t\t"model" may point to a GAN or an Autoencoder checkpoint file.'
              '\n\t\tthe keyword "kw_conditions" will be ignored since the conditions are taken from the checkpoint file.'
              '\n\t\tthe keyword "kw_channel" will be ignored since the samples are already sorted channel-wise.'
              '\n\t\tthe samples are drawn evenly from the saved samples to show the training progress.'
              '\n\t1.2 If the keyword "data" is given'
              '\n\t\t"data" must point to a csv-file.'
              '\n\t\tthe keyword "kw_conditions" must be given to identify the condition column(s).'
              '\n\t\tthe samples will be drawn randomly from the dataset.')
        print('2.\tThe keyword "loss" works only with the keyword "checkpoint".')
        print('3.\tThe keyword "average" averages either'
              '\n\tall the samples (if no condition is given)'
              '\n\talong each combination of conditions that is given. The conditions are shown in the legend.')
        print('4.\tWhen using the keywords "pca" or "tsne" the keyword "comp_data" must be defined.'
              '\n\tExcept for the case "model" is given and the checkpoint file is an Autoencoder file.'
              '\n\tIn this case, the comparison dataset (original data) is taken from the Autoencoder file directly.')
        print('5.\tThe keyword "channel_plots" can be used to enhace the visualization.'
              '\n\tThis way, the channels are shown in different subplots along the columns.')
        print('6.\tThe keyword "channel_index" can be defined to plot only a subset of channels.'
              '\n\tIf the keyword "channel_index" is not given, all channels are plotted.'
              '\n\tSeveral channels can be defined list-like e.g., "channel_index=0,4,6,8".')
        self.end_line()


class HelperGenerateSamples(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tThe keyword "file" carries some special features:'
              '\n\t1.1 It is possible to give only a file instead of a whole file path'
              '\n\t\tIn this case, the default path is "trained_models"'
              '\n\t1.2 The specified file must be a checkpoint file which contains the generator state dict and its '
              '\n\t    corresponding configuration dict')
        print('2.\tThe keyword "sequence_length_total" defines the length of the generated sequences'
              '\n\tThe default value is -1, which means that the max sequence length is chosen'
              '\n\tThe max sequence length is determined by the used training data set given by the configuration dict')
        print('3.\tThe keyword "condition" defines the condition for the generator:'
              '\n\t3.1 Hereby, the value can be either a scalar or a comma-seperated list of scalars e.g., "condition=1,3.234,0"'
              '\n\t    Current implementation: The single elements must be numeric'
              '\n\t    The length of the condition must be equal to the "n_condition" parameter in the configuration dict'
              '\n\t3.2 The value -1 means that the condition is chosen randomly'
              '\n\t    This works currently only for binary conditions. ')
        print('4.\tThe keyword "num_samples_parallel" defines the number of generated samples in one batch'
              '\n\tThis parameter should be set according to the processing power of the used machine'
              '\n\tEspecially, the generation of large number of sequences can be boosted by increasing this parameter')


def default_inputs_training_gan():
    kw_dict = {
        'ddp': [bool, 'Activate distributed training', False, 'Distributed training is active'],
        'seed': [bool, 'Set seed for reproducibility', None, 'Manual seed: '],
        'n_epochs': [int, 'Number of epochs', 100, 'Number of epochs: '],
        'batch_size': [int, 'Batch size', 128, 'Batch size: '],
        'sample_interval': [int, 'Interval of epochs between saving samples', 100, 'Sample interval: '],
        'hidden_dim': [int, 'Hidden dimension of the GAN components', 16, 'Hidden dimension: '],
        'num_layers': [int, 'Number of layers of the GAN components', 4, 'Number of layers: '],
        'patch_size': [int, 'Patch size of the divided sequence', 20, 'Patch size: '],
        'discriminator_lr': [float, 'Learning rate for the discriminator', 0.0001, 'Discriminator learning rate: '],
        'generator_lr': [float, 'Learning rate for the generator', 0.0001, 'Generator learning rate: '],
        'data': [str, 'Path to a dataset', os.path.join('data', 'gansEEGTrainingData.csv'), 'Dataset: '],
        'checkpoint': [str, 'Path to a pre-trained GAN', '', 'Using pre-trained GAN: '],
        'autoencoder': [str, 'Path to an autoencoder', '', 'Using autoencoder: '],
        'kw_conditions': [str, '** Conditions to be used', '', 'Conditions: '],
        'kw_time': [str, 'Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time', 'Time', 'Time label: '],
        'kw_channel': [str, 'Keyword to detect used channels', '', 'Channel label: '],
        'save_name': [str, 'Name to save model', '', 'Model save name: '],
    }

    return kw_dict


def default_inputs_training_autoencoder():
    kw_dict = {
        'ddp': [bool, 'Activate distributed training', False, 'Distributed training is active'],
        'load_checkpoint': [bool, 'Load a pre-trained AE', False, 'Loading a trained autoencoder model'],
        'seed': [bool, 'Set seed for reproducibility', None, 'Manual seed: '],
        'data': [str, 'Path to the dataset', os.path.join('data', 'gansEEGTrainingData.csv'), 'Dataset: '],
        'checkpoint': [str, 'Path to a pre-trained AE', '', 'Using pre-trained AE: '],
        'save_name': [str, 'Name to save model', '', 'Model save name: '],
        'target': [str, 'Target dimension (channel, time, full) to encode; full is recommended for multi-channel data;', 'full', 'Target: '],
        'kw_time': [str, 'Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time', 'Time', 'Time label: '],
        'kw_channel': [str, 'Keyword to detect used channels', '', 'Channel label: '],
        'activation': [str, 'Activation function of the AE decoder; Options: [relu, leakyrelu, sigmoid, tanh, linear]', 'sigmoid', 'Activation function: '],
        'channels_out': [int, 'Size of the encoded channels', 10, 'Encoded channels size: '],
        'time_out': [int, 'Size of the encoded timeseries', 10, 'Encoded time series size: '],
        'n_epochs': [int, 'Number of epochs to train for', 100, 'Number of epochs: '],
        'batch_size': [int, 'Batch size', 128, 'Batch size: '],
        'sample_interval': [int, 'Interval of epochs between saving samples', 100, 'Sample interval: '],
        'hidden_dim': [int, 'Hidden dimension of the transformer', 256, 'Hidden dimension: '],
        'num_layers': [int, 'Number of layers of the transformer', 2, 'Number of layers: '],
        'num_heads': [int, 'Number of heads of the transformer', 8, 'Number of heads: '],
        'train_ratio': [float, 'Ratio of training data to total data', 0.8, 'Training ratio: '],
        'learning_rate': [float, 'Learning rate of the AE', 0.0001, 'Learning rate: '],
    }
    return kw_dict


def default_inputs_training_vae():
    kw_dict = {
        'load_checkpoint': [bool, 'Load a pre-trained AE', False, 'Loading a trained autoencoder model'],
        'data': [str, 'Path to the dataset', os.path.join('data', 'ganTrialElectrodeERP_p500_e1_SS100_Run00.csv'), 'Dataset: '], #TODO: REMOVE THIS
        'path_checkpoint': [str, 'Path to a trained model to continue training', os.path.join('trained_ae', 'checkpoint.pt'), 'Checkpoint: '],
        'save_name': [str, 'Name to save model', None, 'Model save name: '],
        'sample_interval': [int, 'Interval of epochs between saving samples', 100, 'Sample interval: '],
        'kw_channel': [str, 'Column name to detect used channels', '', 'Channel label: '],
        'kw_conditions': [str, '** Conditions to be used', 'Condition', 'Conditions: '],
        'kw_time': [str, 'Keyword for the time step of the dataset', 'Time', 'Keyword for the time step of the dataset: '],
        # TODO: check for which components of VAE the parameter 'activation' applies
        'activation': [str, 'Activation function of the AE components; Options: [relu, leakyrelu, sigmoid, tanh, linear]', 'tanh', 'Activation function: '],
        'n_epochs': [int, 'Number of epochs to train for', 1000, 'Number of epochs: '],
        'batch_size': [int, 'Batch size', 128, 'Batch size: '],
        'hidden_dim': [int, 'Hidden dimension of the network', 256, 'Hidden dimension: '],
        'encoded_dim': [int, 'Encoded dimension of mu and sigma', 25, 'Encoded dimension: '],
        'learning_rate': [float, 'Learning rate of the VAE', 3e-4, 'Learning rate: '],
        'kl_alpha': [float, 'Weight of the KL divergence in loss', 0.00005, 'KL alpha: '],
        }
    return kw_dict


def default_inputs_visualize():
    kw_dict = {
        'loss': [bool, 'Plot training loss', False, 'Plotting training loss'],
        'average': [bool, 'Average over all samples to get one averaged curve (per condition, if any is given)', False, 'Averaging over all samples'],
        'pca': [bool, 'Use PCA to reduce the dimensionality of the data', False, 'Using PCA'],
        'tsne': [bool, 'Use t-SNE to reduce the dimensionality of the data', False, 'Using t-SNE'],
        'spectogram': [bool, 'Use spectogram to visualize the frequency distribution of the data', False, 'Using spectogram'],
        'fft': [bool, 'Use a FFT-histogram to visualize the frequency distribution of the data', False, 'Using FFT-Hist'],
        'channel_plots': [bool, 'Plot each channel in a separate column', False, 'Plotting each channel in a separate column'],
        'model': [str, 'Use samples from checkpoint file', '', 'Using samples from model/checkpoint file (.pt)'],
        'data': [str, 'Use samples from csv-file', '', 'Using samples from csv-file'],
        'comp_data': [str, 'Path to a csv dataset for comparison; comparison only for t-SNE or PCA;', os.path.join('data', 'ganAverageERP.csv'), 'Comparison dataset: '],
        'kw_conditions': [str, '** Conditions to be used', '', 'Conditions: '],
        'kw_time': [str, 'Keyword to detect the time steps of the dataset; e.g. if [Time1, Time2, ...] -> use Time', 'Time', 'Time label: '],
        'kw_channel': [str, 'Keyword to detect used channels', '', 'Channel label: '],
        'n_samples': [int, 'Total number of samples to be plotted', 0, 'Number of plotted samples: '],
        'channel_index': [int, '**Index of the channel to be plotted; If -1, all channels will be plotted;', -1, 'Index of the channels to be plotted: '],
        'tsne_perplexity': [int, 'Perplexity of t-SNE', 40, 'Perplexity of t-SNE: '],
        'tsne_iterations': [int, 'Number of iterations of t-SNE', 1000, 'Number of iterations of t-SNE: '],
    }

    return kw_dict


def default_inputs_checkpoint_to_csv():
    kw_dict = {
        'model': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt'), 'Model: '],
        'key': [str, '** Key of the checkpoint file to be saved; "losses" or "generated_samples"', 'generated_samples', 'Key: '],
    }

    return kw_dict


def default_inputs_generate_samples():
    kw_dict = {
        'seed': [bool, 'Set seed for reproducibility', None, 'Manual seed: '],
        'model': [str, 'File which contains the trained model and its configuration', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
        'save_name': [str, 'File where to store the generated samples; If None, then checkpoint name is used', '', 'Saving generated samples to file: '],
        'kw_time': [str, 'Keyword for the time step of the dataset; to determine the sequence length', 'Time', 'Keyword for the time step of the dataset: '],
        'sequence_length': [int, 'total sequence length of generated sample; if -1, then sequence length from training dataset', -1, 'Total sequence length of a generated sample: '],
        'num_samples_total': [int, 'total number of generated samples', 1000, 'Total number of generated samples: '],
        'num_samples_parallel': [int, 'number of samples generated in parallel', 50, 'Number of samples generated in parallel: '],
        'conditions': [int, '** Specific numeric conditions', None, 'Conditions: '],
    }

    return kw_dict


def default_inputs_get_gan_config():
    kw_dict = {
        'model': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
    }

    return kw_dict


def return_list(string_list, separator=','):
    """Splits the list-like string by the separator and convert the string to the correct type.
    Current implemented types are: int, float, bool, str.
    These types are determined individually for each element of the list."""

    ls = []

    if not isinstance(string_list, str):
        string_list = str(string_list)

    for x in string_list.split(separator):
        if x.isnumeric() or '-1' in x:
            if '.' in x:
                ls.append(float(x))
            else:
                ls.append(int(x))
        elif x == 'True' or x == 'False':
            ls.append(bool(x))
        else:
            ls.append(x)
    return ls


def parse_arguments(arguments, kw_dict=None, file=None):
    """Parses the given arguments and returns a dictionary with the parsed arguments.
    If the argument is not given, the default value is used.
    If the argument is given, it is checked if it is of the correct type.
    If the argument is a list, it is split by the comma and the elements are converted to the correct type."""
    # print(arguments)

    # get default system arguments
    system_args = {}
    if file is not None:
        if file == 'visualize_main.py':
            system_args = default_inputs_visualize()
            helper = HelperVisualize(system_args)
        elif file == 'gan_training_main.py':
            system_args = default_inputs_training_gan()
            helper = HelperMain(system_args)
        elif file == 'generate_samples_main.py':
            system_args = default_inputs_generate_samples()
            helper = HelperGenerateSamples(system_args)
        elif file == 'autoencoder_training_main.py':
            system_args = default_inputs_training_autoencoder()
            helper = HelperAutoencoder(system_args)
        elif file == 'vae_training_main.py':
            system_args = default_inputs_training_vae()
            helper = HelperVAE(system_args)
        else:
            raise ValueError(f'File {file} not recognized.')
    else:
        system_args = kw_dict
        helper = Helper(kw_dict)

    default_args = {}
    for key, value in system_args.items():
        # value = [type, description, default value]
        if '**' in value[1]:
            # return list if '**' in argument's description
            default_args[key] = return_list(value[2])
        else:
            default_args[key] = value[2]

    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')

    for arg in arguments:
        if '.py' not in arg:
            if arg == 'help':
                # get help
                helper.print_table()
                helper.print_help()
                exit()
            elif arg in system_args.keys():
                # process boolean argument
                print(system_args[arg][3])
                default_args[arg] = True
            elif '=' in arg:
                # process keyword argument
                kw = arg.split('=')
                if kw[0] in system_args.keys():
                    if '**' in system_args[kw[0]][1] or ',' in kw[1]:
                        # process list if either
                        # the keyword's description contains '**' or the argument contains a comma
                        kw[1] = return_list(kw[1])
                    else:
                        # check type of argument
                        if system_args[kw[0]][0] == int:
                            kw[1] = int(kw[1])
                        elif system_args[kw[0]][0] == float:
                            kw[1] = float(kw[1])
                        elif system_args[kw[0]][0] == bool:
                            kw[1] = bool(kw[1])
                        elif system_args[kw[0]][0] == str:
                            kw[1] = str(kw[1])
                        elif system_args[kw[0]][0] is None:
                            kw[1] = None
                    print(system_args[kw[0]][3] + str(kw[1]))
                    default_args[kw[0]] = kw[1]
                else:
                    raise ValueError(
                        f'Keyword {kw[0]} not recognized. Please use the keyword "help" to see the available arguments.')
            else:
                raise ValueError(f'Keyword {arg} not recognized. Please use the keyword "help" to see the available arguments.')

    return default_args


if __name__ == '__main__':
    helper = HelperVisualize('gan_training_main.py', default_inputs_visualize())
    helper.print_table()
    helper.print_help()
