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
                if type(value[2]) != value[0]:
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
            '2.\tUse "ddp=True" to activate distributed training. '
            '\n\tOnly if multiple GPUs are available for one node.'
            '\n\tAll available GPUs are used for training.'
            '\n\tEach GPUs trains on the whole dataset. '
            '\n\tHence, the number of training epochs is multiplied by the number of GPUs')
        print(
            '3.\tIf you want to load a pre-trained GAN, you can use the following command:'
            '\n\tpython gan_training_main.py load_checkpoint; The default file is "trained_models/checkpoint.pt"'
            '\n\tIf you want to use an other file, you can use the following command:'
            '\n\t\tpython gan_training_main.py load_checkpoint path_checkpoint="path/to/file.pt"')
        print(
            '4.\tIf you want to use a different dataset, you can use the following command:'
            '\n\tpython gan_training_main.py path_dataset="path/to/file.csv"'
            '\n\tThe default dataset is "data/ganAverageERP.csv"')
        print(
            '5.\tThe keyword "sequence_length" has two different meanings based on the keyword "windows_slices":'
            '\n\t5.1 "windows_slices" is set to "False": '
            '\n\t\tThe keyword "sequence_length" defines the length of the taken sequence from the dataset.'
            '\n\t\tHereby, only the first {sequence_length} data points are taken from each sample.'
            '\n\t\tThe default value is -1, which means that the whole sequence is taken.'
            '\n\t5.2 "windows_slices" is set to "True": '
            '\n\t\tThe keyword "sequence_length" defines the length of a single window taken from the dataset.'
            '\n\t\tHereby, a sample from the dataset is sliced into windows of length "sequence_length".'
            '\n\t\tEach window is then used as a single sample.'
            '\n\t\tThe samples are taken by moving the window with a specific stride (=5) over the samples.')
        print(
            '5.\tHave in mind to change the keyword patch_size if you use another value for the keyword sequence_length.'
            '\n\tThe condition sequence_length % patch_size == 0 must be fulfilled.'
            '\n\tOtherwise the sequence will be padded with zeros until the condition is fulfilled.')
        print(
            '6.\tThe keyword "seq_len_generated" describes the length of the generated sequences.'
            '\n\t6.1 The condition "seq_len_generated" <= "sequence_length" must be fulfilled.'
            '\n\t6.2 The generator works in the following manner:'
            '\n\t\tThe generator gets a sequence of length ("sequence_length"-"seq_len_generated") as a condition (input).'
            '\n\t\tThe generator generates a sequence of length "seq_len_generated" as output which is used as the '
            '\n\t\tsubsequent part of the input sequence.'
            '\n\t6.3 If ("seq_len_generated" == "sequence_length"):'
            '\n\t\tThe generator does not get any input sequence but generates an arbitrary sequence of length "sequence_length".'
            '\n\t\tArbitrary means hereby that the generator does not get any conditions on previous data points.')
        self.start_line()
        self.end_line()


class HelperVisualize(Helper):
    def __init__(self, kw_dict):
        super().__init__(kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tThe keyword "file" carries some special features:'
              '\n\t1.1 It is possible to give only a file instead of a whole file path.'
              '\n\t\tIn this case, the default path is specified regarding the following keywords:'
              '\n\t\t"checkpoint"\t->\tpath = "trained_models"'
              '\n\t\t"experiment"\t->\tpath = "data"'
              '\n\t\t"csv_file"\t->\tpath = "generated_samples"'
              '\n\t1.2 Specification of the keyword "file":'
              '\n\t\tThe default file works only in combination with the keyword "checkpoint".'
              '\n\t\tIn any other case, the default file must be specified with a compatible file name.')
        print('2.\tIf the keyword "starting_row" is given, the dataset will start from the given row.'
              '\n\tThis utility is useful to skip early training stage samples.'
              '\n\tThe value can also be negative to specify the last n entries '
              '\n\te.g. "starting_row=-100":\tThe last 100 samples of the dataset are used.')
        print('4.\tThe keyword "plot_losses" works only with the keyword "checkpoint".')
        print('5.\tWhen using the keywords "pca" or "tsne" the "training_file" can be defined.'
              '\n\tThe legend corresponds to the plotted samples according to:'
              '\n\t\tred\t->\tsamples from "training_file"'
              '\n\t\tblue\t->\tsamples from "file"'
              '\n\tTherefore, the blue samples can also correspond to samples from an experiment file.')
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
        'load_checkpoint': [bool, 'Load a pre-trained GAN', False, 'Using a pre-trained GAN'],
        'train_gan': [bool, 'Train a GAN', True, 'Training a GAN'],
        'filter_generator': [bool, 'Use low-pass filter on the generator output', False, 'Using a low-pass filter on the GAN output'],
        'windows_slices': [bool, 'Use sliding windows instead of whole sequences', False, 'Using windows slices'],
        'n_epochs': [int, 'Number of epochs', 100, 'Number of epochs: '],
        'batch_size': [int, 'Batch size', 128, 'Batch size: '],
        'patch_size': [int, 'Patch size', 20, 'Patch size: '],
        'sequence_length': [int, 'Used length of the datasets sequences; If None, then the whole sequence is used', -1, 'Total sequence length: '],
        'seq_len_generated': [int, 'Length of the generated sequence', -1, 'Generated sequence length: '],
        'sample_interval': [int, 'Interval of epochs between saving samples', 10, 'Sample interval: '],
        'learning_rate': [float, 'Learning rate of the GAN', 0.0001, 'Learning rate: '],
        'path_dataset': [str, 'Path to the dataset', os.path.join('data', 'gansEEGTrainingData.csv'), 'Dataset: '],
        'path_checkpoint': [str, 'Path to the checkpoint', os.path.join('trained_models', 'checkpoint.pt'), 'Checkpoint: '],
        'ddp_backend': [str, 'Backend for the DDP-Training; "nccl" for GPU; "gloo" for CPU;', 'nccl', 'DDP backend: '],
        'conditions': [str, '** Conditions to be used', 'Condition', 'Conditions: '],
        'kw_timestep_dataset': [str, 'Keyword for the time step of the dataset', 'Time', 'Keyword for the time step of the dataset: '],
        'multichannel': [bool, 'Multi-channel training regime', False, 'Multi-channel training regime: '],
        'chan_label': [str, 'Channel column name when using multichannel', 'Electrode', 'Channels: ']
    }

    return kw_dict


def default_inputs_training_classifier():
    kw_dict = {
        'experiment': [bool, "Use experiment's samples as dataset", False, "Use experiment's samples as dataset"],
        'generated': [bool, 'Use generated samples as dataset', False, 'Use generated samples as dataset'],
        'ddp': [bool, 'Activate distributed training', False, 'Distributed training is active'],
        'testing': [bool, 'Only test. No training', False, 'Testing only'],
        'load_checkpoint': [bool, 'Load a pre-trained GAN', False, 'Using a pre-trained GAN'],
        'n_epochs': [int, 'Number of epochs', 100, 'Number of epochs: '],
        'batch_size': [int, 'Batch size', 128, 'Batch size: '],
        'patch_size': [int, 'Patch size', 20, 'Patch size: '],
        'sequence_length': [int, 'Used length of the datasets sequences; If None, then the whole sequence is used', -1, 'Total sequence length: '],
        'sample_interval': [int, 'Interval of epochs between saving samples', 1000, 'Sample interval: '],
        'learning_rate': [float, 'Learning rate of the GAN', 0.0001, 'Learning rate: '],
        'path_dataset': [str, 'Path to the dataset', os.path.join('data', 'ganAverageERP_len100.csv'), 'Dataset: '],
        'path_test': [str, 'Path to the test dataset if using generated samples', 'None', 'Test dataset: '],
        'path_checkpoint': [str, 'Path to the checkpoint', os.path.join('trained_classifier', 'checkpoint.pt'), 'Checkpoint: '],
        'path_critic': [str, 'Path to the trained critic', os.path.join('trained_models', 'checkpoint.pt'), 'Critic: '],
        'ddp_backend': [str, 'Backend for the DDP-Training; "nccl" for GPU; "gloo" for CPU;', 'nccl', 'DDP backend: '],
        'conditions': [str, '** Conditions to be used', 'Condition', 'Conditions: '],
        'kw_timestep_dataset': [str, 'Keyword for the time step of the dataset', 'Time', 'Keyword for the time step of the dataset: '],
    }

    return kw_dict


def default_inputs_visualize():
    kw_dict = {
        'file': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
        'training_file': [str, 'Path to the original data', os.path.join('data', 'ganAverageERP.csv'), 'Training dataset: '],
        'kw_timestep_dataset': [str, 'Keyword for the time step of the dataset', 'Time', 'Keyword for the time step of the dataset: '],
        'conditions': [str, '** Conditions to be used', 'Condition', 'Conditions: '],
        'checkpoint': [bool, 'Use samples from training checkpoint file', False, 'Using samples from checkpoint file'],
        'experiment': [bool, 'Use samples from experimental data', False, 'Using samples from experimental data'],
        'csv_file': [bool, 'Use samples from csv-file', False, 'Using samples from csv-file'],
        'plot_losses': [bool, 'Plot training losses', False, 'Plotting training losses'],
        'averaged': [bool, 'Average over all samples to get one averaged curve', False, 'Averaging over all samples'],
        'pca': [bool, 'Use PCA to reduce the dimensionality of the data', False, 'Using PCA'],
        'tsne': [bool, 'Use t-SNE to reduce the dimensionality of the data', False, 'Using t-SNE'],
        'spectogram': [bool, 'Use spectogram to visualize the frequency distribution of the data', False, 'Using spectogram'],
        'fft_hist': [bool, 'Use a FFT-histogram to visualize the frequency distribution of the data', False, 'Using FFT-Hist'],
        'save': [bool, 'Save the generated plots in the directory "plots" instead of showing them', False, 'Saving plots'],
        # 'save_data': [bool, 'Save the curve data in the directory "plots" as a csv file', False, 'Saving data'],
        'bandpass': [bool, 'Use bandpass filter from models.TtsGeneratorFiltered.filter() on samples', False, 'Using low-pass filter'],
        # 'mvg_avg': [bool, 'Use moving average filter on samples', False, 'Using moving average filter'],
        # 'mvg_avg_window': [int, 'Window of moving average filter', 100, 'Window of moving average filter: '],
        'n_conditions': [int, 'Number of conditions as first columns in data', 1, 'Number of conditions: '],
        'n_samples': [int, 'Total number of samples to be plotted', 10, 'Number of plotted samples: '],
        'batch_size': [int, 'Number of samples in one plot', 10, 'Number of samples in one plot: '],
        'starting_row': [int, 'Starting row of the dataset', 0, 'Starting to plot from row: '],
        'tsne_perplexity': [int, 'Perplexity of t-SNE', 40, 'Perplexity of t-SNE: '],
        'tsne_iterations': [int, 'Number of iterations of t-SNE', 1000, 'Number of iterations of t-SNE: '],
    }

    return kw_dict


def default_inputs_checkpoint_to_csv():
    kw_dict = {
        'file': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
        'key': [str, '** Key of the checkpoint file to be saved; "losses" or "generated_samples"', 'generated_samples', 'Key: '],
    }

    return kw_dict


def default_inputs_generate_samples():
    kw_dict = {
        'file': [str, 'File which contains the trained model and its configuration', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
        'path_samples': [str, 'File where to store the generated samples; If None, then checkpoint name is used', 'None', 'Saving generated samples to file: '],
        'kw_timestep_dataset': [str, 'Keyword for the time step of the dataset; to determine the sequence length', 'Time', 'Keyword for the time step of the dataset: '],
        'sequence_length_total': [int, 'total sequence length of generated sample; if -1, then sequence length from training dataset', -1, 'Total sequence length of a generated sample: '],
        'num_samples_total': [int, 'total number of generated samples', 1000, 'Total number of generated samples: '],
        'num_samples_parallel': [int, 'number of samples generated in parallel', 50, 'Number of samples generated in parallel: '],
        'conditions': [int, '** Specific condition; -1 -> random condition (only for binary condition)', -1, 'Conditions: '],
        'average': [int, 'Average over n latent variables to get an averaged one', 1, 'Average over n latent variables: '],
        'all_cond_per_z': [bool, 'PRELIMINARY; ONLY FOR SINGLE BINARY CONDITION; Generate all conditions per latent variable', False, 'Generating all conditions per latent variable'],
    }

    return kw_dict


def default_inputs_get_gan_config():
    kw_dict = {
        'file': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt'), 'File: '],
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
