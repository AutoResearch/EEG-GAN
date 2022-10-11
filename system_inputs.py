"""This file shows which inputs can be given to gan_training_main.py from the command line."""
import os


class Helper:
    def __init__(self, file, kw_dict):
        self.file = file
        self.kw_dict = kw_dict

    def print_table(self):
        # Define what to print in table

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
        print(f"INPUT HELP - These are the inputs that can be given to {self.file} from the command line")
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
              '\n\tWhitespaces are not allowed between a keyword and its value.')
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
    def __init__(self, file, kw_dict):
        super().__init__(file, kw_dict)

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
            '\n\t\tAnother dictionary is saved as "state_dict_{n_epochs}ep_{timestamp}.pt".'
            '\n\t\tThis file contains everything the checkpoint file contains, plus the generated samples.')
        print(
            '2.\tUse "ddp=True" to activate distributed training. But only if multiple GPUs are available for one node.')
        print(
            '3.\tIf you want to load a pre-trained GAN, you can use the following command:'
            '\n\tpython gan_training_main.py load_checkpoint; The default file is "../trained_models/checkpoint.pt"'
            '\n\tIf you want to use an other file, you can use the following command:'
            '\n\t\tpython gan_training_main.py load_checkpoint path_checkpoint="path/to/file.pt"')
        print(
            '4.\tIf you want to use a different dataset, you can use the following command:'
            '\n\tpython gan_training_main.py path_dataset="path/to/file.csv"'
            '\n\tThe default dataset is "../data/ganAverageERP.csv"')
        print(
            '5.\tThe keyword "sequence_length" has to different meanings based on the keyword "windows_slices":'
            '\n\t5.1 "windows_slices" is set to "False": '
            '\n\t\tThe keyword "sequence_length" defines the length of the taken sequence from the dataset.'
            '\n\t\tHereby, only the first "sequence_length" data points are taken from each sample.'
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
            '\n\t\tArbitrary means that the generator does not get any conditions on previous data points.')
        self.start_line()
        self.end_line()


class HelperVisualize(Helper):
    def __init__(self, file, kw_dict):
        super().__init__(file, kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tThe keyword "file" carries some special features:'
              '\n\t1.1 It is possible to give only a file instead of a whole file path.'
              '\n\t\tIn this case, the default path is specified regarding the following keywords:'
              '\n\t\t"checkpoint"\t->\tpath = "trained_models"'
              '\n\t\t"generate"\t->\tpath = "trained_models"'
              '\n\t\t"experiment"\t->\tpath = "data"'
              '\n\t\t"csv_file"\t->\tpath = "generated_samples"'
              '\n\t1.2 Specification of the keyword "file":'
              '\n\t\tThe default file works only in combination with the keywords "checkpoint" or "generate".'
              '\n\t\tIn any other case, the default file must be specified with a compatible file name.')
        print('2.\tThe keyword "n_samples" has two slightly different meanings regarding the following keywords:'
              '\n\t2.1 "experiment", "checkpoint" or "csv_file":'
              '\n\t\tThe "n_samples" samples are drawn linearly from the dataset'
              '\n\t2.2 "generate":'
              '\n\t\t "n_samples" samples are generated through the loaded model')
        print('3.\tIf the keyword "starting_row" is given, the dataset will start from the given row.'
              '\n\tThis utility is useful to skip early training stage samples.'
              '\n\tThe value can also be negative to specify the last n entries '
              '\n\te.g. "starting_row=-100":\tThe last 100 samples of the dataset are used.')
        print('4.\tFollowing filters can be applied to the drawn or generated samples:'
              '\n\t4.1 keyword "bandpass":\tThe bandpass filter from "models.TtsGeneratorFiltered" is applied'
              '\n\t4.2 keyword "mvg_avg":\tA moving average filter with the window length "mvg_avg_window" is applied')
        print('5.\tThe keyword "plot_losses" works only with the keyword "checkpoint".'
              '\n\tHereby, it is advisable to use the filter "mvg_avg" to get a more comprehensible visualization.')
        self.end_line()


def default_inputs_main():
    kw_dict = {
        'ddp': [bool, 'Activate distributed training', False],
        'load_checkpoint': [bool, 'Load a pre-trained GAN', False],
        'train_gan': [bool, 'Train a GAN', True],
        'filter_generator': [bool, 'Use low-pass filter on the generator output', False],
        'windows_slices': [bool, 'Use sliding windows instead of whole sequences', False], ####
        'n_epochs': [int, 'Number of epochs', 100], ####
        'batch_size': [int, 'Batch size', 128], ####
        'patch_size': [int, 'Patch size', 15],
        'sequence_length': [int, 'Used length of the datasets sequences; If None, then the whole sequence is used', -1],
        'seq_len_generated': [int, 'Length of the generated sequence', -1],
        'sample_interval': [int, 'Interval of batches between saving samples', 1000], ####
        'n_conditions': [int, 'Number of conditions', 1],
        'learning_rate': [float, 'Learning rate of the GAN', 0.0001],
        'path_dataset': [str, 'Path to the dataset', os.path.join('data', 'ganAverageERP.csv')], ####
        'path_checkpoint': [str, 'Path to the checkpoint', os.path.join('trained_models', 'checkpoint.pt')],
        'ddp_backend': [str, 'Backend for the DDP-Training; "nccl" for GPU; "gloo" for CPU;', 'nccl'],
    }

    for key, value in kw_dict.items():
        if type(value[2]) != value[0]:
            raise TypeError(f'Default value of {key} is not of given type {value[0]}. Please correct the default value.')

    return kw_dict


def default_inputs_visualize():
    # generate, experimental, load_file, filename, n_samples, batch_size, starting_row, n_conditions, filter, \
    # mvg_avg, mvg_avg_window
    kw_dict = {
        'file': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt')],
        'checkpoint': [bool, 'Use samples from training checkpoint file', False],
        'generate': [bool, 'Use generator to create samples to visualize', False],
        'experiment': [bool, 'Use samples from experimental data', False],
        'csv_file': [bool, 'Use samples from csv-file', False],
        'plot_losses': [bool, 'Plot training losses', False],
        'save': [bool, 'Save the generated plots in the directory "plots" instead of showing them', False],
        'bandpass': [bool, 'Use bandpass filter on samples', False],
        'mvg_avg': [bool, 'Use moving average filter on samples', False],
        'mvg_avg_window': [int, 'Window of moving average filter', 100],
        'n_conditions': [int, 'Number of conditions as first columns in data', 1],
        'n_samples': [int, 'Total number of samples to be plotted', 10],
        'batch_size': [int, 'Number of samples in one plot', 10],
        'starting_row': [int, 'Starting row of the dataset', 0],
    }

    return kw_dict


def default_inputs_checkpoint_to_csv():
    kw_dict = {
        'file': [str, 'File to be used', os.path.join('trained_models', 'checkpoint.pt')],
        'key': [str, 'Key of the checkpoint file to be saved; The other option is "losses"; Can be also given like: =key_1,key_2', 'generated_samples'],
    }

    return kw_dict


if __name__ == '__main__':
    helper = HelperVisualize('gan_training_main.py', default_inputs_visualize())
    helper.print_table()
    helper.print_help()
