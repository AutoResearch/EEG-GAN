import numpy as np
import pandas as pd
np.random.seed(42)

#Load data
print('Loading data...')
data = np.genfromtxt(f'data/ERPCORE/N400/Full Datasets/erpcore_N400_full.csv', delimiter=',', skip_header=1)
print('Data loaded.')

#####################################################
## Randomly select training, validation and test sets
#####################################################

#Determine participant IDs for each dataset
p_IDs = np.unique(data[:,0])
np.random.shuffle(p_IDs)

validation_IDs = p_IDs[:10] #10 participants
test_IDs = p_IDs[10:20] #10 participants
train_IDs = p_IDs[20:] #20 participants

print(f'Validation IDs: {validation_IDs}, size: {len(validation_IDs)}')
print(f'Test IDs: {test_IDs}, size: {len(test_IDs)}')
print(f'Train IDs: {train_IDs}, size: {len(train_IDs)}')

#Split data into training, validation and test sets
validation_data = data[np.isin(data[:,0], validation_IDs)]
test_data = data[np.isin(data[:,0], test_IDs)]
train_data = data[np.isin(data[:,0], train_IDs)]

#Create headers
headers = ['Participant_ID', 'Condition', 'Trial', 'Electrode']
for t in range(128):
    headers.append(f'Time{t+1}')

#Save datasets
validation_data = pd.DataFrame(validation_data)
validation_data.to_csv(f'data/ERPCORE/N400/Validation and Test Datasets/erpcore_N400_validation.csv', header=headers, index=False)

test_data = pd.DataFrame(test_data)
test_data.to_csv(f'data/ERPCORE/N400/Validation and Test Datasets/erpcore_N400_test.csv', header=headers, index=False)

train_data = pd.DataFrame(train_data)
train_data.to_csv(f'data/ERPCORE/N400/Validation and Test Datasets/erpcore_N400_train.csv', header=headers, index=False)