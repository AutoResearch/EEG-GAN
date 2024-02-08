
import pandas as pd
import matplotlib.pyplot as plt
import torch

#### Define parameters ####
gan_filename = 'aegan_ep8000_p100_e2_enc25-1.pt'
data_filename = 'ganTrialElectrodeERP_p100_e2_len100.csv'

#### Load generated data ####
syn_filename = gan_filename.replace('.pt','')
c0_syn = pd.read_csv(f'generated_samples/{syn_filename}_c0.csv')
c1_syn = pd.read_csv(f'generated_samples/{syn_filename}_c1.csv')
syn_data_index = 2

#### Load empirical data ####
c_emp = pd.read_csv(f'data/{data_filename}')
c0_emp = c_emp[c_emp['Condition']==0]
c1_emp = c_emp[c_emp['Condition']==1]
emp_data_index = 4

#### Load GAN ####
state_dict = torch.load(f"trained_models/{gan_filename}", map_location=torch.device('cpu'))

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
    #Empirical
    ax[0,int(electrode-1)].plot(c0_emp[c0_emp['Electrode']==electrode].mean()[emp_data_index:], label='C0')
    ax[0,int(electrode-1)].plot(c1_emp[c1_emp['Electrode']==electrode].mean()[emp_data_index:], label='C1')
    ax[0,int(electrode-1)].set_title(f'Empirical (E: {int(electrode)})')
    ax[0,int(electrode-1)].get_xaxis().set_visible(False)
    ax[0,int(electrode-1)].get_yaxis().set_visible(False)
    ax[0,int(electrode-1)].spines[['right', 'top']].set_visible(False)

    #Synthetic
    ax[1,int(electrode-1)].plot(c0_syn[c0_syn['Electrode']==electrode].mean()[syn_data_index:], label='C0')
    ax[1,int(electrode-1)].plot(c1_syn[c1_syn['Electrode']==electrode].mean()[syn_data_index:], label='C1')
    ax[1,int(electrode-1)].set_title(f'Synthetic (E: {int(electrode)})')
    ax[1,int(electrode-1)].get_xaxis().set_visible(False)
    ax[1,int(electrode-1)].get_yaxis().set_visible(False)
    ax[1,int(electrode-1)].spines[['right', 'top']].set_visible(False)
plt.show()