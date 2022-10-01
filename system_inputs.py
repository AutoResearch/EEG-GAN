"""This file shows which inputs can be given to gan_training_main.py from the command line."""


def default_inputs():
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
        'path_dataset': [str, 'Path to the dataset', 'data/ganAverageERP.csv'],
        'path_checkpoint': [str, 'Path to the checkpoint', 'trained_models/checkpoint.pt'],
    }

    for key, value in kw_dict.items():
        if type(value[2]) != value[0]:
            raise TypeError(f'Default value of {key} is not of given type {value[0]}. Please correct the default value.')

    return kw_dict


def print_table(kw_dict):
    # Define what to print in table

    values = list(kw_dict.keys())
    types = [kw_dict[key][0] for key in values]
    descriptions = [kw_dict[key][1] for key in values]
    defaults = [kw_dict[key][2] for key in values]

    # Get longest string length of each column
    max_len = [0, 0, 0, 0]
    for i in range(len(kw_dict)):
        max_len[0] = max(max_len[0], len(values[i]))
        max_len[1] = max(max_len[1], len(str(types[i])))
        max_len[2] = max(max_len[2], len(descriptions[i]))
        max_len[3] = max(max_len[3], len(str(defaults[i])))

    # Print table
    print('\n----------------------------------------------------------------------------------------------------------------------------------------------------')
    print("INPUT HELP - These are the inputs that can be given to gan_training_main.py from the command line")
    print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')

    # print header of table
    print(
        f'{"Input":<{max_len[0]}} | {"Type":<{max_len[1]}} | {"Description":<{max_len[2]}} | {"Default value":<{max_len[3]}}')
    print('----------------------------------------------------------------------------------------------------------------------------------------------------')

    # Print values of table
    for i in range(len(values)):
        # define spacing for each column
        spacing = [' ' * (max_len[0] - len(values[i])), ' ' * (max_len[1] - len(str(types[i]))),
                   ' ' * (max_len[2] - len(descriptions[i])), ' ' * (max_len[3] - len(str(defaults[i])))]
        # print row
        print(
            f'{values[i]}{spacing[0]} | {str(types[i])}{spacing[1]} | {descriptions[i]}{spacing[2]} | {defaults[i]}{spacing[3]}')

    # print end of table
    print('\n----------------------------------------------------------------------------------------------------------------------------------------------------\n')


def print_help():
    """Print help message for gan_training_main.py regarding special features of the system arguments."""
    print('\n----------------------------------------------------------------------------------------------------------------------------------------------------')
    print("QUICK HELP - These are the special features of the system arguments:")
    print('----------------------------------------------------------------------------------------------------------------------------------------------------\n')
    print('1. Use "ddp=True" to activate distributed training. '
          '\n\tBut only if multiple GPUs are available for one node.')
    print('1. If you want to load a pre-trained GAN, you can use the following command:'
          '\n\tpython gan_training_main.py load_checkpoint=True; The default file is "trained_models/checkpoint.pt"'
          '\n\tIf you want to use an other file, you can use the following command:'
          '\n\t\tpython gan_training_main.py load_checkpoint=True path_checkpoint="path/to/file.pt"')
    print('2. If you want to use a different dataset, you can use the following command:'
          '\n\tpython gan_training_main.py path_dataset="path/to/file.csv"; The default dataset is "data/ganAverageERP.csv"')
    print('3. If you want to use only the first n time steps of the datasets sequences, you can use the following command:'
          '\n\tpython gan_training_main.py sequence_length=n; The default value is 90')
    print('4. If you use windows slices (windows_slices=True), the keyword sequence_length describes the width of the sequence windows instead.')
    print('5 Have in mind to change the keyword patch_size if you use another value for the keyword sequence_length.'
          '\n\tThe condition sequence_length % patch_size == 0 must be fulfilled. '
          '\n\tOtherwise the sequence will be padded with zeros until the condition is fulfilled.')
    print('6. The keyword seq_len_generated describes the length of the generated sequences. The default value is 10.'
          '\n\tThe condition seq_len_generated <= sequence_length must be fulfilled.'
          '\n\tThe generator works in the following manner:'
          '\n\t\tThe generator gets a sequence of length (sequence_length-seq_len_generated) as input.'
          '\n\t\tThe generator generates a sequence of length seq_len_generated as output which is the subsequent part of the input sequence.'
          '\n\t\tIf (seq_len_generated == sequence_length), the generator does not get any input sequence but generates an arbitrary sequence of length seq_len_generated.')
    print('\n----------------------------------------------------------------------------------------------------------------------------------------------------\n')

if __name__ == '__main__':
    kw = default_inputs()
    print_table(kw)
    print_help()
