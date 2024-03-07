
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

#### User input ####
gan_path = 'trained_models/Reinforcement Learning'
files = [f for f in os.listdir(gan_path) if os.path.isfile(os.path.join(gan_path, f))]

for gan_filename in files:

    #### Load generated data ####
    syn_filename = gan_filename.replace('.pt','')
    c0_syn = pd.read_csv(f'generated_samples/Reinforcement Learning/{syn_filename}_c0.csv')
    c1_syn = pd.read_csv(f'generated_samples/Reinforcement Learning/{syn_filename}_c1.csv')
    syn_data_index = 2

    #### Load empirical data ####
    data_path = 'data/Reinforcement Learning/Training Datasets'
    data_filename = gan_filename.replace('aegan_ep2000','ganTrialElectrodeERP').replace('.pt','.csv')
    c_emp = pd.read_csv(f'{data_path}/{data_filename}')
    c0_emp = c_emp[c_emp['Condition']==0]
    c1_emp = c_emp[c_emp['Condition']==1]
    emp_data_index = 4

    #### Load GAN ####
    state_dict = torch.load(f"trained_models/Reinforcement Learning/{gan_filename}", map_location=torch.device('cpu'))

    #### Plot losses ####
    plt.plot(state_dict['generator_loss'], alpha=.75, label='Generator')
    plt.plot(state_dict['discriminator_loss'], alpha=.75, label='Discriminator')
    plt.legend()
    plt.show()

    #### Plot individual trials ####
    '''
    fig, ax = plt.subplots(len(c0_syn['Electrode'].unique()))

    for trial in range(100):
        ax[0].plot(c0_syn[c0_syn['Electrode']==1].iloc[trial][syn_data_index:], alpha=.03, label='C0', color='C0')
        ax[1].plot(c0_syn[c0_syn['Electrode']==2].iloc[trial][syn_data_index:], alpha=.03, label='C0', color='C0')
        
        ax[0].plot(c1_syn[c1_syn['Electrode']==1].iloc[trial][syn_data_index:], alpha=.03, label='C1', color='C1')
        ax[1].plot(c1_syn[c1_syn['Electrode']==2].iloc[trial][syn_data_index:], alpha=.03, label='C1', color='C1')
    plt.show()
    '''

    #### Plot data ####
    fig, ax = plt.subplots(2,len(c0_syn['Electrode'].unique()))

    for electrode in c0_syn['Electrode'].unique():

        if '_e1_' in gan_filename:
            ax[int(electrode-1)].plot(c0_emp[c0_emp['Electrode']==electrode].mean()[emp_data_index:], label='C0')
            ax[int(electrode-1)].plot(c1_emp[c1_emp['Electrode']==electrode].mean()[emp_data_index:], label='C1')
            ax[int(electrode-1)].set_title(f'Empirical (E: {int(electrode)})')
            ax[int(electrode-1)].get_xaxis().set_visible(False)
            ax[int(electrode-1)].get_yaxis().set_visible(False)
            ax[int(electrode-1)].spines[['right', 'top']].set_visible(False)
            ax[int(electrode-1)].set_title(gan_filename.split('/')[-1])

            #Synthetic
            ax[int(electrode)].plot(c0_syn[c0_syn['Electrode']==electrode].mean()[syn_data_index:], label='C0')
            ax[int(electrode)].plot(c1_syn[c1_syn['Electrode']==electrode].mean()[syn_data_index:], label='C1')
            ax[int(electrode)].set_title(f'Synthetic (E: {int(electrode)})')
            ax[int(electrode)].get_xaxis().set_visible(False)
            ax[int(electrode)].get_yaxis().set_visible(False)
            ax[int(electrode)].spines[['right', 'top']].set_visible(False)

        else:
            #Empirical
            ax[0,int(electrode-1)].plot(c0_emp[c0_emp['Electrode']==electrode].mean()[emp_data_index:], label='C0')
            ax[0,int(electrode-1)].plot(c1_emp[c1_emp['Electrode']==electrode].mean()[emp_data_index:], label='C1')
            ax[0,int(electrode-1)].set_title(f'Empirical (E: {int(electrode)})')
            ax[0,int(electrode-1)].get_xaxis().set_visible(False)
            ax[0,int(electrode-1)].get_yaxis().set_visible(False)
            ax[0,int(electrode-1)].spines[['right', 'top']].set_visible(False)
            ax[0,0].set_title(gan_filename.split('/')[-1])

            #Synthetic
            ax[1,int(electrode-1)].plot(c0_syn[c0_syn['Electrode']==electrode].mean()[syn_data_index:], label='C0')
            ax[1,int(electrode-1)].plot(c1_syn[c1_syn['Electrode']==electrode].mean()[syn_data_index:], label='C1')
            ax[1,int(electrode-1)].set_title(f'Synthetic (E: {int(electrode)})')
            ax[1,int(electrode-1)].get_xaxis().set_visible(False)
            ax[1,int(electrode-1)].get_yaxis().set_visible(False)
            ax[1,int(electrode-1)].spines[['right', 'top']].set_visible(False)
    plt.show()