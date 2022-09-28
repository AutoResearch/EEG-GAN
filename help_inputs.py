if __name__ == '__main__':
    """This file shows which inputs can be given to main.py from the command line."""

    # Define what to print in table
    values = ['n_epochs', 'sequence_length', 'seq_len_generated', 'trained_gan', 'train_gan', 'windows_slices', 'patch_size',
              'batch_size', 'learning_rate', 'sample_interval', 'n_conditions', 'path_dataset']
    types = ['int', 'int', 'int', 'bool', 'bool', 'bool', 'int', 'int', 'float', 'int', 'int', 'str']
    description = ['Number of epochs', 'Total sequence length', 'Sequence length to generate', 'Use pre-trained GAN',
                   'Train GAN', 'Use window slices', 'Patch size of transformer', 'Batch size', 'Learning rate',
                   'Save sample every n batches', 'Number of conditions', 'Path to dataset']
    default = [100, 90, 10, True, True, False, 15, 32, 0.0001, 100, 1, 'data/ganAverageERP.csv']

    # Get longest string length of each column
    max_len = [0, 0, 0, 0]
    for i in range(len(values)):
        max_len[0] = max(max_len[0], len(values[i]))
        max_len[1] = max(max_len[1], len(types[i]))
        max_len[2] = max(max_len[2], len(description[i]))
        max_len[3] = max(max_len[3], len(str(default[i])))

    # Print table
    print('\n--------------------------------------------------------------------------')
    print("These are the inputs that can be given to main.py from the command line")
    print('--------------------------------------------------------------------------\n')

    # print header of table
    print(f'{"Input":<{max_len[0]}} | {"Type":<{max_len[1]}} | {"Description":<{max_len[2]}} | {"Default value":<{max_len[3]}}')
    print('--------------------------------------------------------------------------')

    # Print values of table
    for i in range(len(values)):
        # define spacing for each column
        spacing = [' ' * (max_len[0] - len(values[i])), ' ' * (max_len[1] - len(types[i])),
                   ' ' * (max_len[2] - len(description[i])), ' ' * (max_len[3] - len(str(default[i])))]
        # print row
        print(f'{values[i]}{spacing[0]} | {types[i]}{spacing[1]} | {description[i]}{spacing[2]} | {default[i]}{spacing[3]}')

    # print end of table
    print('\n--------------------------------------------------------------------------\n')