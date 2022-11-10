import os
import sys

import numpy as np
import pandas as pd
import torch

from helpers import system_inputs

if __name__ == '__main__':
    """Bring the safed generated curves from a checkpoint file into a readable csv-format"""
    # sys.argv = ["file=sd_len100_fullseq_1100ep.pt", "key=losses"]
    default_args = system_inputs.parse_arguments(sys.argv, kw_dict=system_inputs.default_inputs_checkpoint_to_csv())

    print('\n-----------------------------------------')
    print("System output:")
    print('-----------------------------------------\n')

    file = default_args['file']
    key = default_args['key']

    if not isinstance(key, list):
        key = [key]

    if not file.endswith('.pt'):
        raise ValueError("Please specify a .pt-file holding a dictionary with the training data.")
    if file.split(os.path.sep)[0] == file:
        # use default path if only file is given
        path = '../trained_models'
        file = os.path.join(path, file)

    print(f'Loading checkpoint from {file}')

    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    filename = file.split(os.path.sep)[-1].split('.')[0]
    path = '../generated_samples'
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
        pd.DataFrame(data).to_csv(os.path.join(path, f), index=False)

        print(f'"{k}" from checkpoint saved as "{filename}_{k}.csv" in directory "generated_samples"')
