###############################################
## LOAD MODULES                              ##
###############################################
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import signal
import scipy
import random as rnd
from IPython.display import clear_output
import time

from helpers.dataloader import Dataloader

###############################################
## USER INPUTS                               ##
###############################################
features = False #WARNING: Do not use! Not adapted for newer data
validation_or_test = 'validation' #'validation' or 'test' set to predict
data_sample_sizes = ['005','010','015','020','030','060','100'] #Which sample sizes to include
electrodes = [1]
synthetic_data_options = [0, 1] #The code will iterate through this list. 0 = empirical classifications, 1 = augmented classifications
classifiers = ['NN', 'SVM', 'LR'] #The code will iterate through this list

###############################################
## FUNCTIONS                                 ##
###############################################

#Define Filter Function
def filterSyntheticEEG(EEG):
    #Bandpass
    w = [x / 100 for x in [0.1, 30]]
    b, a = signal.butter(4, w, 'band')
    
    #Notch
    b_notch, a_notch = signal.iirnotch(60, 30, 500)

    #Process
    tempFilteredEEG = [signal.filtfilt(b, a, EEG[trial,:]) for trial in range(len(EEG))]
    filteredEEG = [signal.filtfilt(b_notch, a_notch, tempFilteredEEG[trial]) for trial in range(len(EEG))]
    
    return filteredEEG

#Define Baseline Function
def baselineCorrect(EEG):
    #Baseline
    baselineRange = [0, 20]

    #process
    baselinedEEG = [(EEG[trial] - (np.mean(EEG[trial][baselineRange[0]:baselineRange[1]]))) for trial in range(len(EEG))]

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
    extractedERP = [np.mean(EEG[trial,startPoint:endPoint]) for trial in range(len(EEG))]

    return np.array(extractedERP)

#Define Delta and Theta extraction function
def extractFFT(EEG):
    
    #Define FFT transformation function
    def runFFT(EEG):
        
        #Determine parameters
        numberDataPoints = len(EEG) #Determine how many datapoints will be transformed per trial and channel
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
        extractedFFT = [np.mean(fftOutput[deltaIndex[0]:deltaIndex[1]]),np.mean(fftOutput[thetaIndex[0]:thetaIndex[1]])]
        
        return extractedFFT
    
    #Transform and extract across trials
    fftFeatures = [runFFT(EEG[trial,:]) for trial in range(len(EEG))]
    
    return np.array(fftFeatures)

#Define feature extraction function
def extractFeatures(EEG):
    erpFeatures = extractERP(EEG) #Extracts Reward Positivity amplitude
    fftFeatures = extractFFT(EEG) #Extracts Delta and Theta power
    eegFeatures = np.transpose(np.vstack((erpFeatures,np.transpose(fftFeatures)))) #Combine features
    
    return eegFeatures

#Average synthetic data function
def averageSynthetic(synData):
    
    #Determine how many trials to average. 50 is in line with participant trial counts per condition
    samplesToAverage = 50

    #Extract each condition
    lossSynData = synData[synData[:,0]==1,:]
    winSynData = synData[synData[:,0]==0,:]

    #Determine points of 
    lossTimeIndices = np.arange(0,lossSynData.shape[0],samplesToAverage)
    winTimeIndices = np.arange(0,winSynData.shape[0],samplesToAverage)
    
    #Average data while inserting condition codes
    newLossSynData = [np.insert(np.mean(lossSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,1) for trialIndex in lossTimeIndices]
    newWinSynData = [np.insert(np.mean(winSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,0) for trialIndex in winTimeIndices]

    #Combine conditions
    avgSynData = np.vstack((np.asarray(newLossSynData),np.asarray(newWinSynData)))
    
    return avgSynData

#Average empirical data function
def averageEEG(participant_IDs, EEG):
    
    #Determine participant IDs
    participants = np.unique(participant_IDs)
    
    #Conduct averaging
    averagedEEG = [] #Initiate variable
    for participant in participants: #Iterate through participant IDs
        for condition in range(2): #Iterate through conditions
            averagedEEG.append(np.mean(EEG[(participant_IDs==participant)&(EEG[:,0,0]==condition),:], axis=0))
            
    return np.array(averagedEEG)

#Define neural network classifier function
def neuralNetwork(X_train, Y_train, x_test, y_test):
    
    #Define search space
    param_grid = [
        {'hidden_layer_sizes': [(50,), (50,50), (50,25,50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter' : [5000]}]

    #Search over search space
    optimal_params = GridSearchCV(
        MLPClassifier(), 
        param_grid, 
        verbose = True,
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
    
###############################################
## CLASSIFICATION                            ##
###############################################
for electrode in electrodes:

    #Base save file names
    augFilename = f'classification/Classification Results/augmentedPredictions_e{electrode}_XX.csv'
    empFilename = f'classification/Classification Results/empiricalPredictions_e{electrode}_XX.csv'

    #Add test tag if test set being used
    if validation_or_test == 'test':
        augFilename = augFilename.split('.csv')[0]+'_TestClassification.csv'
        empFilename = empFilename.split('.csv')[0]+'_TestClassification.csv'

    #Display parameters
    print('data Sample Sizes: ' )
    print(data_sample_sizes)
    print('Classification Data: ' + validation_or_test)
    print('Augmented Filename: ' + augFilename)
    print('Empirical Filename: ' + empFilename) 

    ###############################################
    ## LOAD VALIDATION/TEST DATA                 ##
    ###############################################

    #Load data 
    if validation_or_test == 'validation':
        test_filename = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode}_validation.csv'
    else:
        test_filename = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode}_test.csv'
    
    dataloader = Dataloader(test_filename, col_label='Condition', channel_label='Electrode')
    EEG_test_data = dataloader.get_data().detach().numpy()
    EEG_meta = pd.read_csv(test_filename).iloc[:,:4]
    EEG_participant_IDs = EEG_meta.loc[EEG_meta['Electrode']==1.0,:]['ParticipantID']

    #Average data
    EEG_test_data = averageEEG(EEG_participant_IDs, EEG_test_data)
        
    #Create outcome variable
    y_test = EEG_test_data[:,0,0]

    #Create test variable
    norm = lambda data: (data-np.min(data)) / (np.max(data) - np.min(data))

    if features:
        x_test = np.array(extractFeatures(EEG_test_data[:,2:])) #Extract features
        x_test = scale(x_test, axis=0) #Scale data within each trial
    else:
        x_test = norm(EEG_test_data[:,1:,:]) #Extract normalized raw EEG

    for classifier in classifiers: #Iterate through classifiers (neural network, support vector machine, logistic regression)
        
        #Determine current filenames
        currentAugFilename = augFilename.replace('XX',classifier)
        currentEmpFilename = empFilename.replace('XX',classifier)
        
        for add_synthetic_data in synthetic_data_options: #Iterate through analyses (empirical, augmented)
            
            #Open corresponding file to write to
            if add_synthetic_data:
                f = open(currentAugFilename, 'a')
            else:
                f = open(currentEmpFilename, 'a')
                    
            for data_sample_size in data_sample_sizes: #Iterate through sample sizes   
                for run in range(5): #Conduct analyses 5 times per sample size
                    
                    ###############################################
                    ## SYNTHETIC PROCESSING                      ##
                    ###############################################
                    if add_synthetic_data:
                        
                        #Load Synthetic Data
                        syn_filename_c0 = f'generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode}_SS{data_sample_size}_Run0{run}_c0.csv'
                        syn_filename_c1 = f'generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode}_SS{data_sample_size}_Run0{run}_c1.csv'

                        dataloader_c0 = Dataloader(syn_filename_c0, col_label='Condition', channel_label='Electrode')
                        dataloader_c1 = Dataloader(syn_filename_c1, col_label='Condition', channel_label='Electrode')

                        syn_data_c0 = dataloader_c0.get_data().detach().numpy()
                        syn_data_c1 = dataloader_c1.get_data().detach().numpy()
                        syn_data = np.concatenate((syn_data_c0,syn_data_c1),axis=0)
                        syn_participant_IDs = np.concatenate((np.repeat(np.arange(1,51,1),50),np.repeat(np.arange(1,51,1),50)))
                        
                        ''' Removed processing for now
                        Extract outcome data
                        syn_outcomes = syn_data[:,0,:]

                        #Process synthetic data
                        processedSynData = filterSyntheticEEG(syn_data[:,1:]) 
                        processedSynData = baselineCorrect(processedSynData)

                        #Create new array for processed synthetic data
                        processedSynData = np.insert(np.asarray(processedSynData),0,syn_outcomes, axis = 1)
                        '''
                        processed_syn_data = syn_data
                        
                        #Average data across trials
                        processed_syn_data = averageEEG(syn_participant_IDs,processed_syn_data)

                        #Extract outcome and feature data
                        syn_outcomes = processed_syn_data[:,0,0] #Extract outcome
                        
                        if features: #If extracting features
                            synPredictors = np.array(extractFeatures(processed_syn_data[:,1:])) #Extract features
                            synPredictors = scale(synPredictors, axis=0) #Scale across samples
                        else:
                            synFeatures = processed_syn_data[:,1:,:] #Extract raw data
                            synPredictors = norm(synFeatures)

                    ###############################################
                    ## EMPIRICAL PROCESSING                      ##
                    ###############################################
                    
                    #Load empirical data
                    temp_filename = f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e{electrode}_SS{data_sample_size}_Run0{run}.csv'
                    
                    dataloader = Dataloader(temp_filename, col_label='Condition', channel_label='Electrode')
                    EEG_data = dataloader.get_data().detach().numpy()
                    EEG_meta = pd.read_csv(temp_filename).iloc[:,:4]
                    EEG_participant_IDs = EEG_meta.loc[EEG_meta['Electrode']==1.0,:]['ParticipantID']

                    #Average data
                    EEG_data = averageEEG(EEG_participant_IDs, EEG_data)
                        
                    #Extract outcome and feature data
                    Y_train = EEG_data[:,0,0]
                    
                    if features: #If extracting features
                        X_train = np.array(extractFeatures(EEG_data[:,2:])) #Extract features
                        X_train = scale(X_train, axis=0) #Scale across samples
                    else:
                        X_train = norm(EEG_data[:,1:,:]) #Extract raw data

                    #Shuffle order of samples
                    train_shuffle = rnd.sample(range(len(X_train)),len(X_train))
                    Y_train = Y_train[train_shuffle]
                    X_train = X_train[train_shuffle,:,:]
                    
                    #Create augmented dataset
                    if add_synthetic_data: #If this is augmented analyses
                        Y_train = np.concatenate((Y_train,syn_outcomes)) #Combine empirical and synthetic outcomes
                        X_train = np.concatenate((X_train,synPredictors)) #Combine empirical and synthetic features
                        
                        #Shuffle order of samples
                        train_shuffle = rnd.sample(range(len(X_train)),len(X_train))
                        X_train = X_train[train_shuffle,:,:]
                        Y_train = Y_train[train_shuffle]

                    #Flatten dataset into vector
                    x_test = np.array([sample.T.flatten() for sample in x_test])
                    X_train = np.array([sample.T.flatten() for sample in X_train])

                    #Report current analyses
                    print(electrode)
                    print(classifier)
                    print('Augmented' if add_synthetic_data else 'Empirical')
                    print('Sample Size: ' + str(int(data_sample_size)))
                    print('Run: ' + str(run))
                
                    ###############################################
                    ## CLASSIFIER                                ##
                    ###############################################
                    
                    #Begin timer
                    startTime = time.time()
                    
                    #Run classifier
                    if classifier == 'NN':
                        optimal_params, predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)
                    elif classifier == 'SVM':
                        optimal_params, predictScore = supportVectorMachine(X_train, Y_train, x_test, y_test)
                    elif classifier == 'LR':
                        optimal_params, predictScore = logisticRegression(X_train, Y_train, x_test, y_test)
                    else:
                        print('Unknown classifier')
                        optimal_params = []
                        predictScore = []
                                
                    ###############################################
                    ## SAVE DATA                                 ##
                    ###############################################
                        
                    #Create list of what to write
                    if add_synthetic_data:
                        toWrite = [str(data_sample_size),str(run),str(0),str(predictScore),str(time.time()-startTime),optimal_params.best_params_]
                    else:
                        toWrite = [str(data_sample_size),str(run),'0',str(predictScore),str(time.time()-startTime),optimal_params.best_params_]

                    #Write data to file
                    for currentWrite in toWrite: #Iterate through write list
                        f.write(str(currentWrite)) #Write current item
                        if not currentWrite==toWrite[-1]: #If not the last item
                            f.write(',') #Add comma
                    f.write('\n') #Creates new line
                    f.flush() #Clears the internal buffer

            f.close() #Close file