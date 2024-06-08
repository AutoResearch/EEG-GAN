#Create a latex table with one column being name and the other column being description
import os
import sys
import numpy as np
import pandas as pd
from eeggan.helpers.system_inputs import default_inputs_training_autoencoder, default_inputs_training_gan, default_inputs_visualize, default_inputs_generate_samples
from collections import OrderedDict


def create_latex_table(kw_dict):
    #Create a dictionary with the keys being the name of the variable and the values being the description
    kw_dict = OrderedDict(sorted(kw_dict.items(), key=lambda x: x[0]))
    kw_dict = {k: [v[1], v[2]] for k, v in kw_dict.items()}
    #Create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(kw_dict, orient='index', columns=[ 'Description', 'Default'])

    return df

def main():
    #Get the default inputs for the training of the autoencoder
    kw_dict = default_inputs_training_autoencoder()
    #Create a latex table from the dictionary
    ae_table = create_latex_table(kw_dict)

    #Get the default inputs for the training of the GAN
    kw_dict = default_inputs_training_gan()
    #Create a latex table from the dictionary
    gan_table = create_latex_table(kw_dict)

    #Get the default inputs for the visualization
    kw_dict = default_inputs_visualize()
    #Create a latex table from the dictionary
    visualize_table = create_latex_table(kw_dict)

    #Get the default inputs for the generation of samples
    kw_dict = default_inputs_generate_samples()
    #Create a latex table from the dictionary
    generate_table = create_latex_table(kw_dict)

    #Load and Save Tables as an .md file
    with open("docs/HowToUse/parameters.md", "w") as f:
        f.write(f"# Function Parameters\n\n")
        f.write(f"This page contains the default parameters for the functions in the eeggan package. The parameters are organized by function and are listed in a table with the parameter name and a description of the parameter.")

        f.write(f"## Autoencoder Training\n\n")
        f.write(f"{ae_table.to_markdown()}\n\n")
        f.write(f"## GAN Training\n\n")
        f.write(f"{gan_table.to_markdown()}\n\n")
        f.write(f"## Visualization\n\n")
        f.write(f"{visualize_table.to_markdown()}\n\n")
        f.write(f"## Generate Samples\n\n")
        f.write(f"{generate_table.to_markdown()}\n\n")

if __name__ == '__main__':
    main()

