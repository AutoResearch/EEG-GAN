import os
import sys

import numpy as np
import pandas as pd
import torch

import system_inputs

if __name__ == '__main__':
    # get default system arguments
    system_args = system_inputs.default_inputs_checkpoint_to_csv()
    default_args = {}
    for key, value in system_args.items():
        # value = [type, description, default value]
        default_args[key] = value[2]

    file, key = None, None

    print('\n-----------------------------------------')
    print('Command line arguments:')
    print('-----------------------------------------\n')
    for arg in sys.argv:
        if '.py' not in arg:
            if arg == 'help':
                helper = system_inputs.Helper('checkpoint_to_csv.py', system_inputs.default_inputs_checkpoint_to_csv())
                helper.print_table()
                helper.print_help()
                exit()
            elif '=' in arg:
                kw = arg.split('=')
                if kw[0] == 'file':
                    print(f'Using file: {kw[1]}')
                    file = kw[1]
                elif kw[0] == 'key':
                    print(f'Using key: {kw[1]}')
                    if ',' in kw[1]:
                        keys = kw[1].split(',')
                        key = [k for k in keys]
                    else:
                        key = [kw[1]]
                else:
                    print(f'Keyword {kw[0]} not recognized. Please use the keyword "help" to see the available arguments.')
            else:
                print(
                    f'Keyword {arg} not recognized. Please use the keyword "help" to see the available arguments.')

    file = default_args['file'] if file is None else file
    key = [default_args['key']] if key is None else key

    if not file.endswith('.pt'):
        raise ValueError("Please specify a .pt-file holding a dictionary with the training data.")
    if file.split(os.path.sep)[0] == file:
        # use default path if only file is given
        path = 'trained_models'
        file = os.path.join(path, file)

    print(f'Loading checkpoint from {file}')

    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    filename = file.split(os.path.sep)[-1].split('.')[0]
    path = 'generated_samples'
    for k in key:
        f = f'{filename}_{k}.csv'
        data = []
        if k == 'losses':
            k_new = ['discriminator_loss', 'generator_loss']
            for kk in k_new:
                data.append(checkpoint[kk])
        else:
            data = checkpoint[k]
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        columns = np.arange(data.shape[1])
        index = np.arange(data.shape[0])
        pd.DataFrame(data).to_csv(os.path.join(path, f))

        print(f'"{k}" from checkpoint saved as "{filename}_{k}.csv" in directory "generated_samples"')
