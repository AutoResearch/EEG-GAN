import pandas as pd
import numpy as np

#############################################################
## ANALYZE RESULTS ##
#############################################################

#Load data
results = pd.read_csv('evaluation/quantitative_evaluation_results.csv').values

#Determine average predict scores per dataset and per column
datasets = np.unique(results[:,0])
emp_scores = []
gan_scores = []
vae_scores = []
emp_var = []
gan_var = []
vae_var = []
for dataset in datasets:
    emp_scores.append(np.mean(results[results[:,0]==dataset,1]))
    gan_scores.append(np.mean(results[results[:,0]==dataset,2]))
    vae_scores.append(np.mean(results[results[:,0]==dataset,3]))

    emp_var.append(np.std(results[results[:,0]==dataset,1], ddof=1))
    gan_var.append(np.std(results[results[:,0]==dataset,2], ddof=1))
    vae_var.append(np.std(results[results[:,0]==dataset,3], ddof=1))

all_scores = np.mean(results[:,1:], axis=0)
all_var = np.std(results[:,1:].astype(float), axis=0, ddof=1)

#Create a combined score where each element is a string of score [std]
emp_report = [f'{score:.0f} ({var:.0f})' for score, var in zip(emp_scores, emp_var)]
gan_report = [f'{score:.0f} ({var:.0f})' for score, var in zip(gan_scores, gan_var)]
vae_report = [f'{score:.0f} ({var:.0f})' for score, var in zip(vae_scores, vae_var)]
all_report = [f'{score:.0f} ({var:.0f})' for score, var in zip(all_scores, all_var)]

#Create a pandas dataframe of results where rows are datasets and columns are methods
results_df = pd.DataFrame(data={'empirical':emp_report, 'GAN':gan_report, 'VAE':vae_report, 'ALL':all_report}, index=datasets)
