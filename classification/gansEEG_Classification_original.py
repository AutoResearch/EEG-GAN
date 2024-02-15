###############################################
## LOAD MODULES                              ##
###############################################
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import signal
import scipy
import random as rnd
import time

###############################################
## USER INPUTS                               ##
###############################################
features = False #Datatype: False = Full Data, True = Features data
validationOrTest = 'validation' #'validation' or 'test' set to predict
dataSampleSizes = ['005','010','015','020','030','060','100'] #Which sample sizes to include
syntheticDataOptions = [0] #The code will iterate through this list. 0 = empirical classifications, 1 = augmented classifications
classifiers = ['NN', 'SVM', 'LR'] #The code will iterate through this list

###############################################
## SETUP                                     ##
###############################################

#Base save file names
augFilename = 'classification/Classification Results/augmentedPredictions_XX_SynP050_Runs8000_Filtered.csv'
empFilename = 'classification/Classification Results/empiricalPredictions_XX_Runs8000.csv'

#Add features tag if applied
if features:
    augFilename = augFilename.split('.csv')[0]+'_Features.csv'
    empFilename = empFilename.split('.csv')[0]+'_Features.csv'

#Add test tag if test set being used
if validationOrTest == 'test':
    augFilename = augFilename.split('.csv')[0]+'_TestClassification.csv'
    empFilename = empFilename.split('.csv')[0]+'_TestClassification.csv'
    
#Display parameters
print('data Sample Sizes: ' )
print(dataSampleSizes)
print('Classification Data: ' + validationOrTest)
print('Augmented Filename: ' + augFilename)
print('Empirical Filename: ' + empFilename) 

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
def averageEEG(EEG):
    
    #Determine participant IDs
    participants = np.unique(EEG[:,0])
    
    #Conduct averaging
    averagedEEG = [] #Initiate variable
    for participant in participants: #Iterate through participant IDs
        for condition in range(2): #Iterate through conditions
            averagedEEG.append(np.mean(EEG[(EEG[:,0]==participant)&(EEG[:,1]==condition),:], axis=0))
            
    return np.array(averagedEEG)

#Cut synthetic data in half function
def cutData(synData):  
    
    #Determine index to cut   
    keepIndex = round((synData.shape[0]*0.5)/2) #Removes half of the data (we generated more samples than we wanted)
    
    #Extract each condition
    lossSynData = synData[synData[:,0]==1,:]
    winSynData = synData[synData[:,0]==0,:]

    #Combine only the kept parts of each dataset
    synData = np.vstack((lossSynData[0:keepIndex,:],winSynData[0:keepIndex,:]))

    return synData

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
## LOAD VALIDATION/TEST DATA                 ##
###############################################

#Load data 
if validationOrTest == 'validation':
    EEGDataTest = np.genfromtxt('data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e1_validation.csv', delimiter=',', skip_header=1)
else:
    EEGDataTest = np.genfromtxt('data/Reinforcement Learning/Validation and Test Datasets/gansTrialERP_len100_TestData.csv', delimiter=',', skip_header=1)

#Average data
EEGDataTest = averageEEG(EEGDataTest)[:,1:]
    
#Create outcome variable
y_test = EEGDataTest[:,0]

#Create test variable
if features:
    x_test = np.array(extractFeatures(EEGDataTest[:,2:])) #Extract features
    x_test = scale(x_test, axis=0) #Scale data within each trial
else:
    x_test = EEGDataTest[:,2:] #Extract raw EEG
    x_test = scale(x_test,axis = 1) #Scale data within each trial

###############################################
## CLASSIFICATION                            ##
###############################################
for classifier in classifiers: #Iterate through classifiers (neural network, support vector machine, logistic regression)
    
    #Determine current filenames
    currentAugFilename = augFilename.replace('XX',classifier)
    currentEmpFilename = empFilename.replace('XX',classifier)
    
    for addSyntheticData in syntheticDataOptions: #Iterate through analyses (empirical, augmented)
        
        #Open corresponding file to write to
        if addSyntheticData:
            f = open(currentAugFilename, 'a')
        else:
            f = open(currentEmpFilename, 'a')
                
        for dataSampleSize in dataSampleSizes: #Iterate through sample sizes   
            for run in range(5): #Conduct analyses 5 times per sample size
                
                ###############################################
                ## SYNTHETIC PROCESSING                      ##
                ###############################################
                if addSyntheticData:
                    
                    #Load Synthetic Data
                    synFilename = '../GANs/GAN Generated Data/filtered_checkpoint_SS' + dataSampleSize + '_Run' + str(run).zfill(2) + '_nepochs8000'+'.csv'
                    synData = np.genfromtxt(synFilename, delimiter=',', skip_header=1)
                    synData = cutData(synData)
                    
                    #Extract outcome data
                    synOutcomes = synData[:,0]

                    #Process synthetic data
                    processedSynData = filterSyntheticEEG(synData[:,1:]) 
                    processedSynData = baselineCorrect(processedSynData)

                    #Create new array for processed synthetic data
                    processedSynData = np.insert(np.asarray(processedSynData),0,synOutcomes, axis = 1)

                    #Average data across trials
                    processedSynData = averageSynthetic(processedSynData)
                    
                    #Extract outcome and feature data
                    synOutomes = processedSynData[:,0] #Extract outcome
                    
                    if features: #If extracting features
                        synPredictors = np.array(extractFeatures(processedSynData[:,1:])) #Extract features
                        synPredictors = scale(synPredictors, axis=0) #Scale across samples
                    else:
                        synFeatures = processedSynData[:,1:] #Extract raw data
                        synPredictors = scale(synFeatures, axis=1) #Scale across timeseries within trials

                ###############################################
                ## EMPIRICAL PROCESSING                      ##
                ###############################################
                
                #Load empirical data
                tempFilename = 'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_SS'+dataSampleSize+ '_Run' + str(run).zfill(2) +'.csv'
                EEGData = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)#[:,1:]
                
                #Average data per participant and condition
                EEGData = averageEEG(EEGData)[:,1:]
                    
                #Extract outcome and feature data
                Y_train = EEGData[:,0]
                
                if features: #If extracting features
                    X_train = np.array(extractFeatures(EEGData[:,2:])) #Extract features
                    X_train = scale(X_train, axis=0) #Scale across samples
                else:
                    X_train = EEGData[:,2:] #Extract raw data
                    X_train = scale(X_train, axis=1) #Scale across timeseries within trials

                #Shuffle order of samples
                trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
                Y_train = Y_train[trainShuffle]
                X_train = X_train[trainShuffle,:]
                
                #Create augmented dataset
                if addSyntheticData: #If this is augmented analyses
                    Y_train = np.concatenate((Y_train,synOutomes)) #Combine empirical and synthetic outcomes
                    X_train = np.concatenate((X_train,synPredictors)) #Combine empirical and synthetic features
                    
                    #Shuffle order of samples
                    trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
                    X_train = X_train[trainShuffle,:]
                    Y_train = Y_train[trainShuffle]

                #Report current analyses
                print(classifier)
                print('Augmented' if addSyntheticData else 'Empirical')
                print('Sample Size: ' + str(int(dataSampleSize)))
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
                if addSyntheticData:
                    toWrite = [str(dataSampleSize),str(run),str(8000),str(predictScore),str(time.time()-startTime),optimal_params.best_params_]
                else:
                    toWrite = [str(dataSampleSize),str(run),'0',str(predictScore),str(time.time()-startTime),optimal_params.best_params_]

                #Write data to file
                for currentWrite in toWrite: #Iterate through write list
                    f.write(str(currentWrite)) #Write current item
                    if not currentWrite==toWrite[-1]: #If not the last item
                        f.write(',') #Add comma
                f.write('\n') #Creates new line
                f.flush() #Clears the internal buffer

        f.close() #Close file