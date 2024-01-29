
import pandas as pd
import matplotlib.pyplot as plt

#### Load generated data ####
filename = 'aegan_p100_ep100_e2'
c0_raw = pd.read_csv(f'generated_samples/{filename}_c0.csv')
c1_raw = pd.read_csv(f'generated_samples/{filename}_c1.csv')
gen_data_index = 2

#### Load empirical data ####
filename_emp = filename.replace('aegan','ganTrialElectrodeERP').replace('_ep1000','').replace('_ep100','').replace('_ep4000','')+'_len100.csv'
c_emp = pd.read_csv(f'data/{filename_emp}')
c0_emp = c_emp[c_emp['Condition']==0]
c1_emp = c_emp[c_emp['Condition']==1]
emp_data_index = 4

#### Plot data ####
fig, ax = plt.subplots(2,len(c0_raw['Electrode'].unique()))

for electrode in c0_raw['Electrode'].unique():
    ax[0,int(electrode-1)].set_title(f'Empirical (E:{electrode})')
    ax[0,int(electrode-1)].plot(c0_emp[c0_emp['Electrode']==electrode].mean()[emp_data_index:], label='C0')
    ax[0,int(electrode-1)].plot(c1_emp[c1_emp['Electrode']==electrode].mean()[emp_data_index:], label='C1')

    ax[1,int(electrode-1)].set_title(f'Synthetic (E:{electrode})')
    ax[1,int(electrode-1)].plot(c0_raw[c0_raw['Electrode']==electrode].mean()[emp_data_index:], label='C0')
    ax[1,int(electrode-1)].plot(c1_raw[c1_raw['Electrode']==electrode].mean()[emp_data_index:], label='C1')
plt.show()