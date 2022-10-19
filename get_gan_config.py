import os
import sys

import torch

if __name__ == "__main__":
    # Default file
    file = os.path.join('trained_models', 'checkpoint.pt')

    for arg in sys.argv:
        if not arg.endswith('.py'):
            if '=' in arg:
                arg = arg.split('=')
                if arg[0] == 'file':
                    file = arg[1]

    if file.split(os.path.sep)[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    state_dict = torch.load(file, map_location='cpu')
    print(f'\nGAN Configuration of {file}:')
    for key, value in state_dict['configuration'].items():
        print(f'\t{key}: {value}')
    print('\n')