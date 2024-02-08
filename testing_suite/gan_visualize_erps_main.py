
import pandas as pd
import matplotlib.pyplot as plt

#### Define parameters ####
gan_type = 'gan' #gan or aegan
participants = 100 #500 or 100
epochs = 100 #100, 1000, or 4000
electrodes = 2 #1, 2, 8

#### Load generated data ####
filename = f'{gan_type}_p{participants}_ep{epochs}_e{electrodes}'
c0_syn = pd.read_csv(f'generated_samples/{filename}_c0.csv')
c1_syn = pd.read_csv(f'generated_samples/{filename}_c1.csv')
gen_data_index = 2

#### Load empirical data ####
filename_emp = filename.replace('aegan','ganTrialElectrodeERP').replace('gan','ganTrialElectrodeERP').replace('_ep1000','').replace('_ep100','').replace('_ep4000','')+'_len100.csv'
c_emp = pd.read_csv(f'data/{filename_emp}')
c0_emp = c_emp[c_emp['Condition']==0]
c1_emp = c_emp[c_emp['Condition']==1]
emp_data_index = 4

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
    ax[1,int(electrode-1)].plot(c0_syn[c0_syn['Electrode']==electrode].mean()[emp_data_index:], label='C0')
    ax[1,int(electrode-1)].plot(c1_syn[c1_syn['Electrode']==electrode].mean()[emp_data_index:], label='C1')
    ax[1,int(electrode-1)].set_title(f'Synthetic (E: {int(electrode)})')
    ax[1,int(electrode-1)].get_xaxis().set_visible(False)
    ax[1,int(electrode-1)].get_yaxis().set_visible(False)
    ax[1,int(electrode-1)].spines[['right', 'top']].set_visible(False)
plt.show()