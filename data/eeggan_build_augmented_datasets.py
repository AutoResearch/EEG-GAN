import pandas as pd
import numpy as np

data_path = f'data/Reinforcement Learning/Training Datasets'
generated_samples_path = f'generated_samples/Reinforcement Learning'
augmented_path = f'data/Reinforcement Learning/Augmented Training Datasets'

for electrode in [1, 2]:
    for sample_size in ['005', '010', '015', '020', '030', '060', '100']:
        for run in [0, 1, 2, 3, 4]:
            empirical_filename = f'{data_path}/ganTrialElectrodeERP_p500_e{electrode}_SS{sample_size}_Run0{run}.csv'
            empirical_data = pd.read_csv(empirical_filename)

            generated_samples_filename_0 = f'{generated_samples_path}/aegan_ep2000_p500_e{electrode}_SS{sample_size}_Run0{run}_c0.csv'
            generated_data_0 = pd.read_csv(generated_samples_filename_0)
            generated_samples_filename_1 = f'{generated_samples_path}/aegan_ep2000_p500_e{electrode}_SS{sample_size}_Run0{run}_c1.csv'
            generated_data_1 = pd.read_csv(generated_samples_filename_1)
            generated_data = pd.concat([generated_data_0, generated_data_1])

            generated_data.insert(0, 'ParticipantID', pd.DataFrame(0, index=np.arange(generated_data.shape[0]), columns=['ParticipantID']))
            generated_data.insert(2, 'Trial', pd.DataFrame(0, index=np.arange(generated_data.shape[0]), columns=['Trial']))
            for timepoint in reversed(range(100)):
                generated_data = generated_data.rename(columns={f'Time{timepoint}': f'Time{timepoint+1}'})

            augmented_data = pd.concat([empirical_data, generated_data])
            augmented_data.to_csv(f'{augmented_path}/augmentedData_p500_e{electrode}_SS{sample_size}_Run0{run}.csv')


