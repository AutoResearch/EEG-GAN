import os
import sys
import torch

from helpers import system_inputs

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
    # sys.argv = ["path_file=trained_models\gan_ddp_8000ep_tanh.pt"]
    main()