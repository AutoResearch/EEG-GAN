"""This file shows which inputs can be given to gan_training_main.py from the command line."""


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
        print("QUICK HELP - These are the special features of the system arguments:")
        self.line()
        print('General information: '
              '\n\tBoolean arguments are given as a single keyword:'
              '\n\t\tSet boolean keyword "test_keyword" to True\t->\tpython file.py test_keyword'
              '\n\tCommand line arguments are given as a keyword followed by an equal sign and the value:'
              '\n\t\tSet command line argument "test_keyword" to "test_value"\t->\tpython file.py test_keyword=test_value'
              '\n\t\tWhitespaces are not allowed between a keyword and its value.')
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
        """Print help message for gan_training_main.py regarding special features of the system arguments."""
        super().print_help()
        print('1. Use "ddp=True" to activate distributed training. '
              '\n\tBut only if multiple GPUs are available for one node.')
        print('1. If you want to load a pre-trained GAN, you can use the following command:'
              '\n\tpython gan_training_main.py load_checkpoint=True; The default file is "trained_models/checkpoint.pt"'
              '\n\tIf you want to use an other file, you can use the following command:'
              '\n\t\tpython gan_training_main.py load_checkpoint=True path_checkpoint="path/to/file.pt"')
        print('2. If you want to use a different dataset, you can use the following command:'
              '\n\tpython gan_training_main.py path_dataset="path/to/file.csv"; The default dataset is "data/ganAverageERP.csv"')
        print(
            '3. If you want to use only the first n time steps of the datasets sequences, you can use the following command:'
            '\n\tpython gan_training_main.py sequence_length=n; The default value is 90')
        print(
            '4. If you use windows slices (windows_slices=True), the keyword sequence_length describes the width of the sequence windows instead.')
        print(
            '5 Have in mind to change the keyword patch_size if you use another value for the keyword sequence_length.'
            '\n\tThe condition sequence_length % patch_size == 0 must be fulfilled. '
            '\n\tOtherwise the sequence will be padded with zeros until the condition is fulfilled.')
        print(
            '6. The keyword seq_len_generated describes the length of the generated sequences. The default value is 10.'
            '\n\tThe condition seq_len_generated <= sequence_length must be fulfilled.'
            '\n\tThe generator works in the following manner:'
            '\n\t\tThe generator gets a sequence of length (sequence_length-seq_len_generated) as input.'
            '\n\t\tThe generator generates a sequence of length seq_len_generated as output which is the subsequent part of the input sequence.'
            '\n\t\tIf (seq_len_generated == sequence_length), the generator does not get any input sequence but generates an arbitrary sequence of length seq_len_generated.')
        self.start_line()
        self.end_line()


class HelperVisualize(Helper):
    def __init__(self, file, kw_dict):
        super().__init__(file, kw_dict)

    def print_help(self):
        super().print_help()
        print('1.\tFor the keyword "file", it is possible to give only a file instead of a whole file path. '
              '\n\t\tIn this case, the default path is specified with regard to the following keywords:'
              '\n\t\t\t"checkpoint"\t->\tpath = "trained_models"'
              '\n\t\t\t"generate"\t->\tpath = "trained_models"'
              '\n\t\t\t"experiment"\t->\tpath = "data"'
              '\n\t\t\t"csv_file"\t->\tpath = "generated_samples"'
              '\n\t\tIf another keyword than "checkpoint" or "generate" is given a file must be specified,'
              '\n\t\tsince the default file is only compatible with these two keywords.')
        print('2.\tIf a dataset is loaded (keywords "experiment", "checkpoint" or "csv_file"),'
              '\n\tthe keyword "n_samples" tells how many samples are drawn linearly from the dataset')
        print('3.\tIf the keyword "starting_row" is given, the dataset will start from the given row.'
              '\n\tThis utility is useful to skip early training stage samples.'
              '\n\tThe value can also be negative.')
        print('4.\tFollowing filters can be applied to the drawn or generated samples:'
              '\n\t\t"bandpass":\tThe bandpass filter from models.TtsGeneratorFiltered is applied'
              '\n\t\t"mvg_avg":\tA moving average filter with the window length "mvg_avg_window" is applied')
        print('5.\tThe keyword "plot_losses" currently works only with the keyword "checkpoint"')
        self.end_line()


def default_inputs_main():
    kw_dict = {
        'ddp': [bool, 'Activate distributed training', False],
        'load_checkpoint': [bool, 'Load a pre-trained GAN', False],
        'train_gan': [bool, 'Train a GAN', True],
        'filter_generator': [bool, 'Use low-pass filter on the generator output', False],
        'windows_slices': [bool, 'Use sliding windows instead of whole sequences', False],
        'n_epochs': [int, 'Number of epochs', 100],
        'batch_size': [int, 'Batch size', 32],
        'patch_size': [int, 'Patch size', 15],
        'sequence_length': [int, 'Used length of the datasets sequences; If None, then the whole sequence is used', 90],
        'seq_len_generated': [int, 'Length of the generated sequence', 10],
        'sample_interval': [int, 'Interval of batches between saving samples', 100],
        'n_conditions': [int, 'Number of conditions', 1],
        'learning_rate': [float, 'Learning rate of the GAN', 0.0001],
        'path_dataset': [str, 'Path to the dataset', r'data\ganAverageERP.csv'],
        'path_checkpoint': [str, 'Path to the checkpoint', r'trained_models\checkpoint.pt'],
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
        'file': [str, 'File to be used', r'trained_models\checkpoint.pt'],
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
        'file': [str, 'File to be used', r'trained_models\checkpoint.pt'],
        'key': [str, 'Key of the checkpoint file to be saved; The other option is "losses"; Can be also given like: =key_1,key_2', 'generated_samples'],
    }

    return kw_dict


if __name__ == '__main__':
    helper = HelperVisualize('gan_training_main.py', default_inputs_visualize())
    helper.print_table()
    helper.print_help()
