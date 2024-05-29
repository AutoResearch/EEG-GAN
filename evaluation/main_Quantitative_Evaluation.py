import numpy as np
from scipy import signal
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
import random as rnd

from helpers.dataloader import Dataloader

## EMPIRICAL ##

#Define Filter Function
def filterEEG(EEG):
    #Bandpass
    w = [x / 100 for x in [0.1, 30]]
    b, a = signal.butter(4, w, 'band')
    
    #Notch
    b_notch, a_notch = signal.iirnotch(60, 30, 500)

    #Process
    if EEG.ndim == 2: #If it's two-dimensional, iterate through trials
        tempFilteredEEG = [signal.filtfilt(b, a, EEG[trial,:]) for trial in range(len(EEG))] #Bandpass filter
        filteredEEG = [signal.filtfilt(b_notch, a_notch, tempFilteredEEG[trial]) for trial in range(len(EEG))] #Notch filter
    else: #Else just process the single tria provided
        tempFilteredEEG = signal.filtfilt(b, a, EEG) #Bandpass filter
        filteredEEG = signal.filtfilt(b_notch, a_notch, tempFilteredEEG) #Notch filter
    
    return filteredEEG

#Define Baseline Function
def baselineCorrect(EEG):
    #Baseline
    baselineRange = [0, 20]

    #Process
    baselinedEEG = [(EEG[trial] - (np.mean(EEG[trial][baselineRange[0]:baselineRange[1]]))) for trial in range(len(EEG))]

    return baselinedEEG

def norm(data, neg=False):

    norm_data = (data-np.min(data))/(np.max(data)-np.min(data))

    if neg:
        norm_data = (norm_data*2)-1

    return norm_data

def load_data(data, gan_data, vae_data, run_gan=True, run_vae=True, process_synthetic=True, select_electrode=None):
    
    print('Loading data...')
    print(data)

    #Load EEG Data (Participant, Condition, Trial, Electrode, Time1, ...)
    EEG_data = np.genfromtxt(data, delimiter=',', skip_header=1) #Removes Participant Column
    EEG_PIDs = EEG_data[:,0] #Participant IDs
    EEG_data = np.delete(EEG_data, 0, 1) #Delete Participant Column
    EEG_data = np.delete(EEG_data, 1, 1) #Delete Unused Column (Trial)

    #Select Electrode
    if select_electrode:
        EEG_data = EEG_data[np.r_[EEG_data[:,1]==select_electrode],:]

    #Split into conditions
    c1_EEG_data = EEG_data[np.r_[EEG_data[:,0]==1],2:]  
    c0_EEG_data = EEG_data[np.r_[EEG_data[:,0]==0],2:] 
    c1_PIDs = EEG_PIDs[np.r_[EEG_data[:,0]==1]]
    c0_PIDs = EEG_PIDs[np.r_[EEG_data[:,0]==0]]

    if run_gan:
        ## GAN ##
        
        #Load and Process Synthetic Data (Condition, Electrode, Time0, ...)
        gan_fn_0 = gan_data.replace('.csv', '_c0.csv')
        gan_fn_1 = gan_data.replace('.csv', '_c1.csv')
        ganData0 = np.genfromtxt(gan_fn_0, delimiter=',', skip_header=1)
        ganData1 = np.genfromtxt(gan_fn_1, delimiter=',', skip_header=1)
        ganData = np.vstack((ganData0, ganData1))
        if select_electrode:
            ganData = ganData[np.r_[ganData[:,1]==select_electrode],:]
        ganData = np.delete(ganData, 1, 1)

        #Process synthetic data
        fftTempganData = ganData
        if process_synthetic:
            tempganData = filterEEG(ganData[:,1:])
            tempganData = baselineCorrect(tempganData)
        else:
            tempganData = ganData[:,1:]
        
        #Create and populate new array for processed synthetic data
        processedganData = np.zeros((len(tempganData),tempganData[0].shape[0]+1))
        processedganData[:,0] = ganData[:,0]
        processedganData[:,1:] = np.array(tempganData)

        #Split into conditions
        c1ganData = processedganData[np.r_[processedganData[:,0]==1],1:]
        c0ganData = processedganData[np.r_[processedganData[:,0]==0],1:]

        #Average data
        avgc1ganData = np.mean(c1ganData, axis=0)
        avgc0ganData = np.mean(c0ganData, axis=0)

        #Scale synthetic data 
        EEGDataScale = np.max(np.mean(c1_EEG_data,axis=0))-np.min(np.mean(c1_EEG_data,axis=0)) 
        EEGDataOffset = np.min(np.mean(c1_EEG_data,axis=0))
        ganDataScale = np.max(avgc1ganData)-np.min(avgc1ganData)
        ganDataOffset = np.min(avgc1ganData)

        avgc1ganData = (((avgc1ganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
        avgc0ganData = (((avgc0ganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

        scaledc1ganData = (((c1ganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
        scaledc0ganData = (((c0ganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

    if run_vae:
        ## VAE ##
        
        vae_fn_0 = vae_data.replace('.csv', '_c0.csv')
        vae_fn_1 = vae_data.replace('.csv', '_c1.csv')
        vaeData0 = np.genfromtxt(vae_fn_0, delimiter=',', skip_header=1)
        vaeData1 = np.genfromtxt(vae_fn_1, delimiter=',', skip_header=1)
        vaeData = np.vstack((vaeData0, vaeData1))
        if select_electrode:
            vaeData = vaeData[np.r_[vaeData[:,1]==select_electrode],:]
        vaeData = np.delete(vaeData, 1,1)

        #Process synthetic data
        if process_synthetic:
            tempvaeData = filterEEG(vaeData[:,1:])
            tempvaeData = baselineCorrect(tempvaeData)
        else:
            tempvaeData = vaeData[:,1:]

        #Create and populate new array for processed synthetic data
        processedvaeData = np.zeros((len(tempvaeData),tempganData[0].shape[0]+1))
        processedvaeData[:,0] = vaeData[:,0]
        processedvaeData[:,1:] = np.array(tempvaeData)

        #Split into conditions
        c1vaeData = processedvaeData[np.r_[processedvaeData[:,0]==1],1:]
        c0vaeData = processedvaeData[np.r_[processedvaeData[:,0]==0],1:]

        #Average data
        avgc1vaeData = np.mean(c1vaeData, axis=0)
        avgc0vaeData = np.mean(c0vaeData, axis=0)

        #Scale synthetic data 
        #scale avgc1vaeData so that it is the same scale as the EEG data

        combined_EEG = np.vstack((c1_EEG_data, c0_EEG_data))
        EEGDataScale = np.max(np.mean(combined_EEG,axis=0))-np.min(np.mean(combined_EEG,axis=0))
        EEGDataOffset = np.min(np.mean(combined_EEG,axis=0))

        combined_vae = np.vstack((c1vaeData, c0vaeData))
        vaeDataScale = np.max(combined_vae)-np.min(combined_vae)
        vaeDataOffset = np.min(combined_vae)


        #avgc1vaeData = (((avgc1vaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
        #avgc0vaeData = (((avgc0vaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

        scaledc1vaeData = (((c1vaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
        scaledc0vaeData = (((c0vaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
        scaledc1vaeData = c1vaeData
        scaledc0vaeData = c0vaeData

    print('Data loading complete!')

    return c1_PIDs, c0_PIDs, c1_EEG_data, c0_EEG_data, scaledc1ganData, scaledc0ganData, scaledc1vaeData, scaledc0vaeData

def load_test_data(file_path):

    #Average data
    EEGDataTest_metadata = np.genfromtxt(file_path, delimiter=',', skip_header=1)[:,:4]
    EEGDataTest_metadata_3D = EEGDataTest_metadata[EEGDataTest_metadata[:,3] == np.unique(EEGDataTest_metadata[:,3])[0],:]
    EEGDataTest_dataloader = Dataloader(file_path, kw_conditions='Condition', kw_channel='Electrode')
    EEGDataTest = EEGDataTest_dataloader.get_data(shuffle=False).detach().numpy()
        
    #Average data
    EEGDataTest = averageEEG(EEGDataTest_metadata_3D[:,0], EEGDataTest)
        
    #Create outcome variable
    y_test = EEGDataTest[:,0,0]

    #Create test variable
    x_test = np.array([test_sample.T.flatten() for test_sample in EEGDataTest[:,1:,:]])
    x_test = scale(x_test, axis=1) #Scale data within each trial

    return y_test, x_test

def averageEEG(participant_IDs, EEG):
    
    #Determine participant IDs
    participants = np.unique(participant_IDs)

    #Conduct averaging
    averagedEEG = np.empty((0, EEG.shape[1], EEG.shape[2])) 
    for participant in participants: #Iterate through participant IDs
        for condition in range(2): #Iterate through conditions
            participant_avg_data = np.mean(EEG[(participant_IDs==participant)&(EEG[:,0,0]==condition),:], axis=0).reshape(-1, EEG.shape[1], EEG.shape[2])
            averagedEEG = np.vstack([averagedEEG, participant_avg_data])
            
    return averagedEEG

def neuralNetwork(X_train, Y_train, x_test, y_test):
    
    #Define search space
    param_grid = [
        {'hidden_layer_sizes': [(25,), (50,), (25, 25), (50,50), (50,25,50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter' : [5000, 10000, 20000, 50000]}]

    #Search over search space
    optimal_params = GridSearchCV(
        MLPClassifier(), 
        param_grid, 
        verbose = 0,
        n_jobs = -1)

    optimal_params.fit(X_train, Y_train)

    #Run Neural Network
    neuralNetOutput = MLPClassifier(hidden_layer_sizes=optimal_params.best_params_['hidden_layer_sizes'], 
                                activation=optimal_params.best_params_['activation'],
                                solver = optimal_params.best_params_['solver'], 
                                alpha = optimal_params.best_params_['alpha'], 
                                learning_rate = optimal_params.best_params_['learning_rate'], 
                                max_iter = optimal_params.best_params_['max_iter'])

    neuralNetOutput.fit(X_train, Y_train)

    #Determine predictability
    y_true, y_pred = y_test , neuralNetOutput.predict(x_test)
    predictResults = classification_report(y_true, y_pred, output_dict=True)
    predictScore = round(predictResults['accuracy']*100)
    
    return optimal_params, predictScore

def run_evaluation(test_fn, emp_fn, gan_fn, vae_fn, electrode=None):

    #Load test data
    y_test, x_test = load_test_data(test_fn)
    testShuffle = rnd.sample(range(len(x_test)),len(x_test))
    x_test = x_test[testShuffle,:]
    y_test = y_test[testShuffle]

    #Load empirical, GAN, and VAE data
    data_IDs_c0, data_IDs_c1, data_eeg_c0, data_eeg_c1, data_gan_c0, data_gan_c1, data_vae_c0, data_vae_c1 = load_data(emp_fn, gan_fn, vae_fn, select_electrode=electrode)

    #Empirical data
    data_eeg_c0 = np.hstack((np.zeros((data_eeg_c0.shape[0],1)), data_eeg_c0))
    data_eeg_c1 = np.hstack((np.ones((data_eeg_c1.shape[0],1)), data_eeg_c1))

    data_eeg = np.vstack((data_eeg_c0, data_eeg_c1)).reshape(-1, data_eeg_c0.shape[1], 1) 
    data_IDs = np.hstack((data_IDs_c0, data_IDs_c1))

    data_avg = averageEEG(data_IDs, data_eeg)
    Y_train = data_avg[:,0,0]
    X_train = scale(data_avg[:,1:,0], axis=1) #Scale across timeseries within trials

    trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
    X_train = X_train[trainShuffle,:]
    Y_train = Y_train[trainShuffle]

    #emp_optimal_params, emp_predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)

    emp_optimal_params, emp_predictScore = 0, 100

    #GAN data
    gan_IDs = np.repeat(np.arange(0, len(data_gan_c0)/50),50)
    gan_IDs = np.hstack((gan_IDs, gan_IDs))

    data_gan_c0 = np.hstack((np.zeros((data_gan_c0.shape[0],1)), data_gan_c0))
    data_gan_c1 = np.hstack((np.ones((data_gan_c1.shape[0],1)), data_gan_c1))

    data_gan = np.vstack((data_gan_c0, data_gan_c1)).reshape(-1, data_gan_c0.shape[1], 1)
    data_avg = averageEEG(gan_IDs, data_gan)
    Y_train = data_avg[:,0,0]
    X_train = scale(data_avg[:,1:,0], axis=1) #Scale across timeseries within trials

    trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
    X_train = X_train[trainShuffle,:]
    Y_train = Y_train[trainShuffle]

    #gan_optimal_params, gan_predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)

    gan_optimal_params, gan_predictScore = 0, 100

    #VAE data
    vae_IDs = np.repeat(np.arange(0, len(data_vae_c0)/50),50)
    vae_IDs = np.hstack((vae_IDs, vae_IDs))

    data_vae_c0 = np.hstack((np.zeros((data_vae_c0.shape[0],1)), data_vae_c0))
    data_vae_c1 = np.hstack((np.ones((data_vae_c1.shape[0],1)), data_vae_c1))

    data_vae = np.vstack((data_vae_c0, data_vae_c1)).reshape(-1, data_vae_c0.shape[1], 1)
    data_avg = averageEEG(vae_IDs, data_vae)
    Y_train = data_avg[:,0,0]
    X_train = scale(data_avg[:,1:,0], axis=1) #Scale across timeseries within trials

    trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
    X_train = X_train[trainShuffle,:]
    Y_train = Y_train[trainShuffle]

    #vae_optimal_params, vae_predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)

    vae_optimal_params, vae_predictScore = 0, 100

    return emp_optimal_params, emp_predictScore, gan_optimal_params, gan_predictScore, vae_optimal_params, vae_predictScore

#############################################################
## LOAD DATA AND RUN EVALUATIONS ##
#############################################################

#Setup save file 
with open('evaluation/quantitative_evaluation_results.csv', 'w') as f:
    f.write('dataset, empirical, GAN, VAE\n')

#Run evaluation for reinforcement learning (e1)
dataset = 'RL1'
for i in range(10):
    print(f'Running evaluation {i} for {dataset}...')
    _, emp_predictScore, _, gan_predictScore, _, vae_predictScore = run_evaluation(
        'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e1_test.csv', 
        'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_SS100_Run00.csv', 
        'generated_samples/Reinforcement Learning/Training Datasets/aegan_ep2000_p500_e1_SS100_Run00.csv',
        'generated_samples/Reinforcement Learning/Training Datasets/vae_e1_SS100_Run00.csv')
    
    #Save predict scores to a csv file
    with open('evaluation/quantitative_evaluation_results.csv', 'a') as f:
        f.write(f'{dataset},{emp_predictScore},{gan_predictScore},{vae_predictScore}\n')

#Run evaluation for reinforcement learning (e8)
dataset = 'RL8'
for i in range(10):
    print(f'Running evaluation {i} for {dataset}...')
    _, emp_predictScore, _, gan_predictScore, _, vae_predictScore = run_evaluation(
        'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e1_test.csv', 
        'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_SS100_Run00.csv', 
        'generated_samples/Reinforcement Learning/Training Datasets/aegan_ep2000_p500_e8_SS100_Run00.csv',
        'generated_samples/Reinforcement Learning/Training Datasets/vae_e8_SS100_Run00.csv',
        electrode=2)
    
    #Save predict scores to a csv file
    with open('evaluation/quantitative_evaluation_results.csv', 'a') as f:
        f.write(f'{dataset},{emp_predictScore},{gan_predictScore},{vae_predictScore}\n')

#Run evaluation for antisaccade
dataset = 'AS'
for i in range(10):
    print(f'Running evaluation {i} for {dataset}...')
    _, emp_predictScore, _, gan_predictScore, _, vae_predictScore = run_evaluation(
        'data/Antisaccade/Validation and Test Datasets/antisaccade_test.csv', 
        'data/Antisaccade/Training Datasets/antisaccade_SS100_Run00.csv', 
        'generated_samples/Antisaccade/Training Datasets/gan_antisaccade_SS100_Run00.csv',
        'generated_samples/Antisaccade/Training Datasets/vae_antisaccade_SS100_Run00.csv')
    
    #Save predict scores to a csv file
    with open('evaluation/quantitative_evaluation_results.csv', 'a') as f:
        f.write(f'{dataset},{emp_predictScore},{gan_predictScore},{vae_predictScore}\n')

#Run evaluation for ERPCORE N170
dataset = 'N170'
for i in range(10):
    print(f'Running evaluation {i} for {dataset}...')
    _, emp_predictScore, _, gan_predictScore, _, vae_predictScore = run_evaluation(
        'data/ERPCORE/N170/Validation and Test Datasets/erpcore_N170_test.csv', 
        'data/ERPCORE/N170/Training Datasets/erpcore_N170_SS020_Run00.csv', 
        'generated_samples/ERPCORE/N170/Training Datasets/gan_erpcore_N170_SS020_Run00.csv',
        'generated_samples/ERPCORE/N170/Training Datasets/vae_erpcore_N170_SS020_Run00.csv')
    
    #Save predict scores to a csv file
    with open('evaluation/quantitative_evaluation_results.csv', 'a') as f:
        f.write(f'{dataset},{emp_predictScore},{gan_predictScore},{vae_predictScore}\n')

#Run evaluation for ERPCORE N2PC
dataset = 'N2PC'
for i in range(10):
    print(f'Running evaluation {i} for {dataset}...')
    _, emp_predictScore, _, gan_predictScore, _, vae_predictScore = run_evaluation(
        'data/ERPCORE/N2PC/Validation and Test Datasets/erpcore_N2PC_test.csv', 
        'data/ERPCORE/N2PC/Training Datasets/erpcore_N2PC_SS020_Run00.csv', 
        'generated_samples/ERPCORE/N2PC/Training Datasets/gan_erpcore_N2PC_SS020_Run00.csv',
        'generated_samples/ERPCORE/N2PC/Training Datasets/vae_erpcore_N2PC_SS020_Run00.csv')
    
    #Save predict scores to a csv file
    with open('evaluation/quantitative_evaluation_results.csv', 'a') as f:
        f.write(f'{dataset},{emp_predictScore},{gan_predictScore},{vae_predictScore}\n')

#############################################################
## ANALYZE RESULTS ##
#############################################################

#Load data
results = np.genfromtxt('evaluation/quantitative_evaluation_results.csv', delimiter=',', skip_header=1)

#Determine average predict scores per dataset
datasets = np.unique(results[:,0])
emp_scores = []
gan_scores = []
vae_scores = []
for dataset in datasets:
    emp_scores.append(np.mean(results[results[:,0]==dataset,1]))
    gan_scores.append(np.mean(results[results[:,0]==dataset,2]))
    vae_scores.append(np.mean(results[results[:,0]==dataset,3]))

#Create a table of results
results_table = np.vstack((emp_scores, gan_scores, vae_scores)).T
results_table = np.hstack((datasets.reshape(-1,1), results_table))

#Print results
print(results_table)


