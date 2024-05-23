from main_plot_functions import *

filename_prefixes = ['classification/Reinforcement Learning/', 
                     'classification/Reinforcement Learning/',
                     'classification/Antisaccade/',
                     'classification/ERPCORE/N170/',
                     'classification/ERPCORE/N2PC/']

save_names = ['figures/Figure S3 - Reinforcement Learning Classification Barplot.png',
                'figures/Figure S4 - Reinforcement Learning e8 Classification Barplot.png',
                'figures/Figure S5 - Antisaccade Classification Barplot.png',
                'figures/Figure S6 - Face Perception Classification Barplot.png',
                'figures/Figure S7 - Visual Search Classification Barplot.png']

x_maxes = [100, 100, 100, 20, 20]

electrodes = [1, 8, 1, 1, 1]

for filename_prefix, save_name, x_max, electrode in zip(filename_prefixes, save_names, x_maxes, electrodes):
    print(filename_prefix, save_name)
    plot_main(filename_prefix=filename_prefix, interactive=False, save_name=save_name, x_max=x_max, electrodes=electrode)
