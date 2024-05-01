import numpy as np
import pandas as pd
np.random.seed(42)

#Load data
print('Loading data...')
data = np.genfromtxt(f'data/antisaccade/Full Datasets/antisaccade_left_full.csv', delimiter=',', skip_header=1)
print('Data loaded.')

#####################################################
## Randomly select training, validation and test sets
#####################################################

#Determine participant IDs for each dataset
p_IDs = np.unique(data[:,0])
np.random.shuffle(p_IDs)

validation_IDs = p_IDs[:49] #49 participants
test_IDs = p_IDs[49:99] #50 participants
train_IDs = p_IDs[99:] #100 participants

print(f'Validation IDs: {validation_IDs}, size: {len(validation_IDs)}')
print(f'Test IDs: {test_IDs}, size: {len(test_IDs)}')
print(f'Train IDs: {train_IDs}, size: {len(train_IDs)}')

#Split data into training, validation and test sets
validation_data = data[np.isin(data[:,0], validation_IDs)]
test_data = data[np.isin(data[:,0], test_IDs)]
train_data = data[np.isin(data[:,0], train_IDs)]

#Create headers
headers = ['Participant_ID', 'Condition', 'Electrode', 'Trial']
for t in range(125):
    headers.append(f'Time{t+1}')

#Save datasets
validation_data = pd.DataFrame(validation_data)
validation_data.to_csv(f'data/antisaccade/Validation and Test Datasets/antisaccade_validation.csv', header=headers, index=False)

test_data = pd.DataFrame(test_data)
test_data.to_csv(f'data/antisaccade/Validation and Test Datasets/antisaccade_test.csv', header=headers, index=False)

train_data = pd.DataFrame(train_data)
train_data.to_csv(f'data/antisaccade/Validation and Test Datasets/antisaccade_train.csv', header=headers, index=False)

#####################################################
## Split training datasets
#####################################################

sample_sizes = [5, 10, 15, 20, 30, 60, 100]
number_runs = 5
train_data = np.array(train_data)

for sample_size in sample_sizes:
    for run in range(number_runs):

        #Determine participant IDs for each dataset
        p_IDs = np.unique(train_data[:,0])
        np.random.shuffle(p_IDs)
        sample_IDs = p_IDs[:sample_size]
        print(f'Sample IDs for SS{str(sample_size).zfill(3)} and Run 0{run}: {sample_IDs}')

        #Extract sample data and save as csv
        sample_data = train_data[np.isin(train_data[:,0], sample_IDs)]
        sample_data = pd.DataFrame(sample_data)
        sample_data.to_csv(f'data/antisaccade/Training Datasets/antisaccade_SS{str(sample_size).zfill(3)}_Run0{run}.csv', header=headers, index=False)