import os
import sys
import torch

# add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
from eeggan.helpers import system_inputs

def main():
    default_args = system_inputs.parse_arguments(sys.argv, kw_dict=system_inputs.default_inputs_get_gan_config())

    file = default_args['model']

    if file.split(os.path.sep)[0] == file:
        # use default path if no path is given
        path = 'trained_models'
        file = os.path.join(path, file)

    state_dict = torch.load(file, map_location='cpu')
    print(f'\nGAN Configuration of {file}:')
    for key, value in state_dict['configuration'].items():
        print(f'\t{key}: {value}')
    print('\n')

if __name__ == "__main__":
    main()