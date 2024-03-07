

import numpy as np 
import matplotlib.pyplot as plt

path = 'generated_samples/Reinforcement Learning'

for ss in ['005','010','015','020','030','060','100']:
    for run in [0, 1, 2, 3, 4]:
        c0 = np.genfromtxt(f"{path}/vae_e1_SS{ss}_Run0{run}_c0.csv", delimiter=',', skip_header=1)[:,2:]
        c1 = np.genfromtxt(f"{path}/vae_e1_SS{ss}_Run0{run}_c1.csv", delimiter=',', skip_header=1)[:,2:]

        plt.plot(np.mean(c0,axis=0))
        plt.plot(np.mean(c1,axis=0))
        plt.title(f'vae_e1_SS{ss}_Run0{run}')
        plt.show()
        plt.close()
