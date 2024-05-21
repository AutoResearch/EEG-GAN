import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import signal
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import scipy

from helpers.dataloader import Dataloader
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder

###############################################
## FUNCTIONS                                 ##
###############################################

#Define Filter Function
def filterSyntheticEEG(EEG):

    #Params
    fs = 83.33
    nyquist = fs/2

    #Bandpass
    w = [x / nyquist for x in [0.1, 30]]
    b, a = signal.butter(4, w, 'band')
    
    #Notch
    b_notch, a_notch = signal.iirnotch(60/nyquist, 30, fs)

    #Process
    filteredEEG = np.empty((0,EEG.shape[1],EEG.shape[2]))
    for trial in range(EEG.shape[0]):
        trialFiltEEGs = np.empty((EEG.shape[1], EEG.shape[2]))
        for electrode in range(EEG.shape[2]):
            trialFiltEEG = signal.filtfilt(b, a, EEG[trial,:,electrode])
            trialFiltEEG = signal.filtfilt(b_notch, a_notch, trialFiltEEG)
            trialFiltEEGs[:,electrode] = trialFiltEEG
        filteredEEG = np.vstack((filteredEEG, trialFiltEEGs.reshape(-1,EEG.shape[1],EEG.shape[2])))

    return filteredEEG

#Define Baseline Function
def baselineCorrect(EEG):
    #Baseline
    baselineRange = [0, 20]

    #process
    baselinedEEG = np.empty((0,EEG.shape[1],EEG.shape[2]))
    for trial in range(EEG.shape[0]):
        trialBaseEEGs = np.empty((EEG.shape[1], EEG.shape[2]))
        for electrode in range(EEG.shape[2]):
            trialEEG = EEG[trial,:,electrode]
            trialBaseEEGs[:,electrode] = trialEEG - (np.mean(trialEEG[baselineRange[0]:baselineRange[1]]))
        baselinedEEG = np.vstack((baselinedEEG, trialBaseEEGs.reshape(-1,EEG.shape[1],EEG.shape[2])))
    
    return baselinedEEG

#Define Reward Positivity extraction function
def extractERP(EEG):
    #Time of interest (ms)
    startTime = 264
    endTime = 356
    
    #Convert to datapoints
    startPoint = round((startTime/12)+16)
    endPoint = round((endTime/12)+17) #Add an extra datapoint so last datapoint is inclusive
    
    #Process
    extractedERP = np.array([np.mean(EEG[trial,startPoint:endPoint,:],axis=0) for trial in range(len(EEG))])

    return extractedERP.reshape(extractedERP.shape[0],1,extractedERP.shape[1])

#Define Delta and Theta extraction function
def extractFFT(EEG):
    
    #Define FFT transformation function
    def runFFT(EEG):
        
        #Determine parameters
        numberDataPoints = EEG.shape[0] #Determine how many datapoints will be transformed per trial and channel
        SR = 83.3333 #Determine sampling rate
        frequencyResolution = SR/numberDataPoints #Determine frequency resolution
        fftFrequencies = np.arange(frequencyResolution,(SR/2),frequencyResolution) #Determine array of frequencies

        #Determine frequencies of interest
        deltaRange = [frequencyResolution,3]
        thetaRange = [4,8]
    
        deltaIndex = [np.where(deltaRange[0]<=fftFrequencies)[0][0], np.where(deltaRange[1]<=fftFrequencies)[0][0]]
        thetaIndex = [np.where(thetaRange[0]<=fftFrequencies)[0][0], np.where(thetaRange[1]<=fftFrequencies)[0][0]]

        #Conduct FFT
        fftOutput = None #Empty variable
        fftOutput = scipy.fft.fft(EEG) #Compute the Fourier transform
        fftOutput = fftOutput/numberDataPoints #Normalize output
        fftOutput = np.abs(fftOutput) #Absolute transformation
        fftOutput = fftOutput[range(int(numberDataPoints/2))] #Extract the one-sided spectrum
        fftOutput = fftOutput*2 #Double values to account for lost values         
        fftOutput = fftOutput**2 #Convert to power
        fftOutput = fftOutput[1:] #Remove DC Offset
        extractedFFT = np.array([np.mean(fftOutput[deltaIndex[0]:deltaIndex[1],:],axis=0),np.mean(fftOutput[thetaIndex[0]:thetaIndex[1],:],axis=0)])
        
        return extractedFFT
    
    #Transform and extract across trials
    fftFeatures = [runFFT(EEG[trial,:,:]) for trial in range(len(EEG))]
    
    return np.array(fftFeatures)

#Define feature extraction function
def extractFeatures(EEG):
    erpFeatures = extractERP(EEG) #Extracts Reward Positivity amplitude
    fftFeatures = extractFFT(EEG) #Extracts Delta and Theta power
    eegFeatures = np.hstack((erpFeatures,fftFeatures))
    
    return eegFeatures

def load_synthetic(synFilename_0, synFilename_1, features, prop_synthetic=None):
                        
    #Load Synthetic Data
    #synFilename = '../GANs/GAN Generated Data/filtered_checkpoint_SS' + dataSampleSize + '_Run' + str(run).zfill(2) + '_nepochs8000'+'.csv'

    Syn0_dataloader = Dataloader(synFilename_0, kw_conditions='Condition', kw_channel='Electrode') #TODO: Condition has 0 in header
    synData_0 = Syn0_dataloader.get_data(shuffle=False).detach().numpy()

    Syn1_dataloader = Dataloader(synFilename_1, kw_conditions='Condition', kw_channel='Electrode')
    synData_1 = Syn1_dataloader.get_data(shuffle=False).detach().numpy()

    if prop_synthetic is not None:
        num_synthetic = int(np.ceil(prop_synthetic*int(synFilename_0.split('SS')[-1].split('_')[0])))
        trials_to_keep = num_synthetic * 50 # 50 trials per participant per condition as defined in averageSynthetic function
        synData_0 = synData_0[:trials_to_keep]
        synData_1 = synData_1[:trials_to_keep]

    synData = np.concatenate((synData_0,synData_1),axis=0)
    
    #Extract outcome data
    synOutcomes = synData[:, 0, 0]

    #Process synthetic data
    processedSynData = filterSyntheticEEG(synData[:,1:,:]) 
    processedSynData = baselineCorrect(processedSynData)

    #Create new array for processed synthetic data
    processedSynData = np.insert(np.asarray(processedSynData),0,synOutcomes.reshape(1,-1,1), axis = 1)

    #Average data across trials
    processedSynData = averageSynthetic(processedSynData)
    
    #Extract outcome and feature data
    syn_Y_train = processedSynData[:,0,0] #Extract outcome
    
    if features: #If extracting features
        syn_X_train = np.array(extractFeatures(processedSynData[:,1:,:])) #Extract features
        syn_X_train = np.array([syn_sample.T.flatten() for syn_sample in syn_X_train])
        syn_X_train = scale(syn_X_train, axis=0) #Scale across samples
    else:
        syn_X_train = np.array([syn_sample.T.flatten() for syn_sample in processedSynData[:,1:,:]])
        syn_X_train = scale(syn_X_train, axis=1) #Scale across timeseries within trials                   

    return syn_Y_train, syn_X_train
    
#Average synthetic data function
def averageSynthetic(synData):
    
    #Determine how many trials to average. 50 is in line with participant trial counts per condition
    samplesToAverage = 50

    #Determine points of 
    participant_IDs = np.repeat(np.arange(synData.shape[0]/samplesToAverage/2), samplesToAverage)
    participant_IDs = np.concatenate((participant_IDs,participant_IDs))
    participants = np.unique(participant_IDs)

    #Average data while inserting condition codes
    averagedEEG = np.empty((0, synData.shape[1], synData.shape[2])) 
    for participant in participants: #Iterate through participant IDs
        for condition in range(2): #Iterate through conditions
            participant_avg_data = np.mean(synData[(participant_IDs==participant)&(synData[:,0,0]==condition),:], axis=0).reshape(-1, synData.shape[1], synData.shape[2])
            averagedEEG = np.vstack([averagedEEG, participant_avg_data])
        
    return averagedEEG

#Average empirical data function
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

#Cut synthetic data in half function
def cutData(synData):  
    
    #Determine index to cut   
    keepIndex = round((synData.shape[0]*0.5)/2) #Removes half of the data (we generated more samples than we wanted)
    
    #Extract each condition
    lossSynData = synData[synData[:,0]==1,:]
    winSynData = synData[synData[:,0]==0,:]

    #Combine only the kept parts of each dataset
    synData = np.v((lossSynData[0:keepIndex,:],winSynData[0:keepIndex,:]))

    return synData

#Define neural network classifier function
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

#Determine support vector machine classifier function
def supportVectorMachine(X_train, Y_train, x_test, y_test):

    # defining parameter range
    param_grid = [
        {'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']}]

    #Search over search space
    optimal_params = GridSearchCV(
        SVC(), 
        param_grid, 
        refit = True, 
        verbose = False)
    
    optimal_params.fit(X_train, Y_train)
    
    #Determine predictability
    SVMOutput = optimal_params.predict(x_test)
    predictResults = classification_report(y_test, SVMOutput, output_dict=True)
    predictScore = round(predictResults['accuracy']*100)
    
    return optimal_params, predictScore
    
#Determine logistic regression classifier function
def logisticRegression(X_train, Y_train, x_test, y_test):
    
    #Define search space            
    param_grid = [
        {'penalty' : ['l1', 'l2'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['liblinear'],
        'max_iter' : [5000, 10000]}]

    #Search over search space
    optimal_params = GridSearchCV(
        LogisticRegression(), 
        param_grid, 
        verbose = False,
        n_jobs = -1)

    optimal_params.fit(X_train, Y_train)
    
    #Determine predictability
    logRegOutput = LogisticRegression(C=optimal_params.best_params_['C'], 
                                penalty=optimal_params.best_params_['penalty'],
                                solver = optimal_params.best_params_['solver'],
                                max_iter = optimal_params.best_params_['max_iter'])
    
    logRegOutput.fit(X_train, Y_train)

    #Determine predictability
    predictScore = round(logRegOutput.score(x_test,y_test)*100)
    
    return optimal_params, predictScore

#Determine random forest classifier function
def randomForest(X_train, Y_train, x_test, y_test):

    # defining parameter range
    param_grid = [
        {'n_estimators': [int(x) for x in np.linspace(start=25, stop=150, num=6)],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [3, 5, 7],
        'criterion': ['gini','entropy']}]

    #Search over search space
    optimal_params = GridSearchCV(
        RandomForestClassifier(), 
        param_grid, 
        refit = True, 
        verbose = False)
    
    optimal_params.fit(X_train, Y_train)
    
    #Determine predictability
    RFOutput = optimal_params.predict(x_test)
    predictResults = classification_report(y_test, RFOutput, output_dict=True)
    predictScore = round(predictResults['accuracy']*100)
    
    return optimal_params, predictScore
    
#Determine support vector machine classifier function
def kNearestNeighbor(X_train, Y_train, x_test, y_test):

    num_samples = X_train.shape[0]
    if num_samples < 11:
        ks = [3, 5, 7]
    elif num_samples < 21:
        ks = [6, 10, 14]
    elif num_samples < 31:
        ks = [9, 15, 21]
    else:
        ks = [12, 18, 28]

    # defining parameter range
    param_grid = [
        {'n_neighbors': ks,
        'weights': ['uniform','distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'leaf_size': [5]
        }]

    #Search over search space
    optimal_params = GridSearchCV(
        KNeighborsClassifier(), 
        param_grid, 
        refit = True, 
        verbose = False)
    
    optimal_params.fit(X_train, Y_train)
    
    #Determine predictability
    SVMOutput = optimal_params.predict(x_test)
    predictResults = classification_report(y_test, SVMOutput, output_dict=True)
    predictScore = round(predictResults['accuracy']*100)
    
    return optimal_params, predictScore
    
def load_test_data(validationOrTest, electrode_number, features):
    if validationOrTest == 'validation':
        EEGDataTest_fn = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_validation.csv'
    else:
        EEGDataTest_fn = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_test.csv'

    #Average data
    EEGDataTest_metadata = np.genfromtxt(EEGDataTest_fn, delimiter=',', skip_header=1)[:,:4]
    EEGDataTest_metadata_3D = EEGDataTest_metadata[EEGDataTest_metadata[:,3] == np.unique(EEGDataTest_metadata[:,3])[0],:]
    EEGDataTest_dataloader = Dataloader(EEGDataTest_fn, kw_conditions='Condition', kw_channel='Electrode')
    EEGDataTest = EEGDataTest_dataloader.get_data(shuffle=False).detach().numpy()
        
    #Average data
    EEGDataTest = averageEEG(EEGDataTest_metadata_3D[:,0], EEGDataTest)
        
    #Create outcome variable
    y_test = EEGDataTest[:,0,0]

    #Create test variable
    if features:
        x_test = np.array(extractFeatures(EEGDataTest[:,1:,:])) #Extract features
        x_test = np.array([test_sample.T.flatten() for test_sample in x_test])
        x_test = scale(x_test, axis=0) #Scale data within each trial
    else:
        x_test = np.array([test_sample.T.flatten() for test_sample in EEGDataTest[:,1:,:]])
        x_test = scale(x_test, axis=1) #Scale data within each trial

    return y_test, x_test