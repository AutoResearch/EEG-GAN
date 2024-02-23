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
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import scipy
import random as rnd
import time
from tqdm import tqdm

from helpers.dataloader import Dataloader
from nn_architecture.ae_networks import TransformerAutoencoder, TransformerDoubleAutoencoder

###############################################
## USER INPUTS                               ##
###############################################
features = False #Datatype: False = Full Data, True = Features data
autoencoder = True #Whether to use autoencoder feature selection
validationOrTest = 'validation' #'validation' or 'test' set to predict
dataSampleSizes = ['005','010','015','020','030','060','100'] #Which sample sizes to include
syntheticDataOptions = [1, 0] #[0, 1, 2] #The code will iterate through this list. 0 = empirical classifications, 1 = augmented classifications, 2 = oversampling classification
classifiers = ['NN', 'SVM', 'LR'] #The code will iterate through this list
electrode_number = 1

###############################################
## SETUP                                     ##
###############################################

#Base save file names
augFilename = f'classification/Classification Results/augmentedPredictions_e{electrode_number}_XX.csv'
empFilename = f'classification/Classification Results/empiricalPredictions_e{electrode_number}_XX.csv'
ovsFilename = f'classification/Classification Results/oversamplingPredictions_e{electrode_number}_XX.csv'

#Add features tag if applied
if features:
    augFilename = augFilename.split('.csv')[0]+'_Features.csv'
    empFilename = empFilename.split('.csv')[0]+'_Features.csv'
    ovsFilename = ovsFilename.split('.csv')[0]+'_Features.csv'

if autoencoder:
    augFilename = augFilename.split('.csv')[0]+'_AE.csv'
    empFilename = empFilename.split('.csv')[0]+'_AE.csv'
    ovsFilename = ovsFilename.split('.csv')[0]+'_AE.csv'

#Add test tag if test set being used
if validationOrTest == 'test':
    augFilename = augFilename.split('.csv')[0]+'_TestClassification.csv'
    empFilename = empFilename.split('.csv')[0]+'_TestClassification.csv'
    ovsFilename = ovsFilename.split('.csv')[0]+'_TestClassification.csv'
    
#Display parameters
print('data Sample Sizes: ' )
print(dataSampleSizes)
print('Classification Data: ' + validationOrTest)
print('Augmented Filename: ' + augFilename)
print('Empirical Filename: ' + empFilename) 
print('Oversampling Filename: ' + ovsFilename)

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

def load_synthetic(electrode_number, dataSampleSize, run, features, autoencoder_filename = None):
                        
    #Load Synthetic Data
    #synFilename = '../GANs/GAN Generated Data/filtered_checkpoint_SS' + dataSampleSize + '_Run' + str(run).zfill(2) + '_nepochs8000'+'.csv'
    synFilename_0 = f"generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c0.csv"
    synFilename_1 = f"generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c1.csv"

    Syn0_dataloader = Dataloader(synFilename_0, col_label='Condition', channel_label='Electrode')
    synData_0 = Syn0_dataloader.get_data(shuffle=False).detach().numpy()

    Syn1_dataloader = Dataloader(synFilename_1, col_label='Condition', channel_label='Electrode')
    synData_1 = Syn1_dataloader.get_data(shuffle=False).detach().numpy()

    synData = np.concatenate((synData_0,synData_1),axis=0)
    
    #Extract outcome data
    synOutcomes = synData[:, 0, 0]

    #Process synthetic data
    processedSynData = filterSyntheticEEG(synData[:,1:,:]) 
    processedSynData = baselineCorrect(processedSynData)

    #Create new array for processed synthetic data
    processedSynData = np.insert(np.asarray(processedSynData),0,synOutcomes.reshape(1,-1,1), axis = 1)

    #Encode data
    if autoencoder_filename != None:
        processedSynData = encode_data(autoencoder_filename, processedSynData)

    #Average data across trials
    processedSynData = averageSynthetic(processedSynData)
    
    #Extract outcome and feature data
    syn_Y_train = processedSynData[:,0,0] #Extract outcome
    
    if features and not autoencoder: #If extracting features
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

def encode_data(autoencoder_filename, data):
    device = torch.device('cpu')
    ae_dict = torch.load(autoencoder_filename, map_location=device)
    if ae_dict['configuration']['target'] == 'channels':
        ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_CHANNELS
        autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
    elif ae_dict['configuration']['target'] == 'time':
        ae_dict['configuration']['target'] = TransformerAutoencoder.TARGET_TIMESERIES
        autoencoder = TransformerAutoencoder(**ae_dict['configuration']).to(device)
    elif ae_dict['configuration']['target'] == 'full':
        autoencoder = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=2).to(device)
        autoencoder.model_1 = TransformerDoubleAutoencoder(**ae_dict['configuration'], training_level=1).to(device)
    else:
        raise ValueError(f"Autoencoder class {ae_dict['configuration']['model_class']} not recognized.")
    consume_prefix_in_state_dict_if_present(ae_dict['model'], 'module.')
    autoencoder.load_state_dict(ae_dict['model'])
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    #Encode data
    if str(data.dtype) == 'float64': data = np.float32(data)
    norm = lambda data: (data-np.min(data)) / (np.max(data) - np.min(data))
    data = np.concatenate((data[:,[0],:], norm(data[:,1:,:])), axis=1)

    time_dim = ae_dict['configuration']['timeseries_out']+1 if ae_dict['configuration']['target'] in [TransformerAutoencoder.TARGET_TIMESERIES, TransformerAutoencoder.TARGET_BOTH] else data.shape[1]
    chan_dim = ae_dict['configuration']['channels_out'] if ae_dict['configuration']['target'] in [TransformerAutoencoder.TARGET_CHANNELS, TransformerAutoencoder.TARGET_BOTH] else data.shape[2]
    ae_dataset = np.empty((data.shape[0], time_dim, chan_dim))
    print('Reconstructing dataset with the autoencoder...')
    for sample in range(data.shape[0]):
        sample_data = data[[sample],1:,:]
        ae_data = autoencoder.encode(torch.from_numpy(sample_data)).detach().numpy()
        ae_dataset[sample,:,:] = np.concatenate((data[sample,0,:].reshape(1,1,-1), ae_data), axis=1)
    
    return ae_dataset

def load_test_data(validationOrTest, electrode_number, features, autoencoder_filename = None):
    if validationOrTest == 'validation':
        EEGDataTest_fn = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_validation.csv'
    else:
        EEGDataTest_fn = f'data/Reinforcement Learning/Validation and Test Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_test.csv.csv'

    #Average data
    EEGDataTest_metadata = np.genfromtxt(EEGDataTest_fn, delimiter=',', skip_header=1)[:,:4]
    EEGDataTest_metadata_3D = EEGDataTest_metadata[EEGDataTest_metadata[:,3] == np.unique(EEGDataTest_metadata[:,3])[0],:]
    EEGDataTest_dataloader = Dataloader(EEGDataTest_fn, col_label='Condition', channel_label='Electrode')
    EEGDataTest = EEGDataTest_dataloader.get_data(shuffle=False).detach().numpy()

    #Encode data
    if autoencoder_filename != None:
        EEGDataTest = encode_data(autoencoder_filename, EEGDataTest)
        
    #Average data
    EEGDataTest = averageEEG(EEGDataTest_metadata_3D[:,0], EEGDataTest)
        
    #Create outcome variable
    y_test = EEGDataTest[:,0,0]

    #Create test variable
    if features and not autoencoder:
        x_test = np.array(extractFeatures(EEGDataTest[:,1:,:])) #Extract features
        x_test = np.array([test_sample.T.flatten() for test_sample in x_test])
        x_test = scale(x_test, axis=0) #Scale data within each trial
    else:
        x_test = np.array([test_sample.T.flatten() for test_sample in EEGDataTest[:,1:,:]])
        x_test = scale(x_test, axis=1) #Scale data within each trial

    return y_test, x_test

###############################################
## LOAD VALIDATION/TEST DATA                 ##
###############################################

#Load data 
y_test, x_test = load_test_data(validationOrTest, electrode_number, features)

###############################################
## CLASSIFICATION                            ##
###############################################
for classifier in classifiers: #Iterate through classifiers (neural network, support vector machine, logistic regression)
    
    #Determine current filenames
    currentAugFilename = augFilename.replace('XX',classifier)
    currentEmpFilename = empFilename.replace('XX',classifier)
    currentOvsFilename = ovsFilename.replace('XX',classifier)
    
    for addSyntheticData in syntheticDataOptions: #Iterate through analyses (empirical, augmented)
        
        #Open corresponding file to write to
        if addSyntheticData==1:
            f = open(currentAugFilename, 'a')
        elif addSyntheticData==0:
            f = open(currentEmpFilename, 'a')
        elif addSyntheticData==2:
            f = open(currentOvsFilename, 'a')
        else:
            print('Analysis index not recognized.')

        for dataSampleSize in dataSampleSizes: #Iterate through sample sizes   
            for run in range(5): #Conduct analyses 5 times per sample size
                
                ###############################################
                ## AUTOENCODE TEST DATA                      ##
                ###############################################
                if autoencoder:
                    if addSyntheticData == 0: #Empirical
                        autoencoder_filename = f'trained_ae/Reinforcement Learning/Embedded/ae_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}.pt'
                    elif addSyntheticData == 1: #Augmented
                        autoencoder_filename = f'trained_ae/Reinforcement Learning/Post-GAN/post_ae_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}.pt'
                    y_test, x_test = load_test_data(validationOrTest, electrode_number, features, autoencoder_filename)
                else:
                    autoencoder_filename = None

                ###############################################
                ## SYNTHETIC PROCESSING                      ##
                ###############################################
                if addSyntheticData == 1:
                    syn_Y_train, syn_X_train = load_synthetic(electrode_number, dataSampleSize, run, features, autoencoder_filename)

                ###############################################
                ## EMPIRICAL PROCESSING                      ##
                ###############################################
                
                #Load empirical data
                tempFilename = f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}.csv'
                EEGData_metadata = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)[:,:4]
                EEGData_metadata_3D = EEGData_metadata[EEGData_metadata[:,3] == np.unique(EEGData_metadata[:,3])[0],:]
                EEGData_dataloader = Dataloader(tempFilename, col_label='Condition', channel_label='Electrode')
                EEGData = EEGData_dataloader.get_data(shuffle=False).detach().numpy()

                #Oversampling analysis
                if addSyntheticData==2:
                    num_participant = np.unique(EEGData_metadata[:,0]).shape[0]
                    participant_IDs = np.unique(EEGData_metadata[:,0])[:50]
                    participant_cycle = 1
                    for pi, participant_ID in enumerate(participant_IDs):
                        participant_index = EEGData_metadata_3D[:,0]==participant_ID
                        participant_EEGData_meta = EEGData_metadata_3D[participant_index,:]
                        participant_EEGData_meta[:,0] = participant_EEGData_meta[:,0] + (1000*participant_cycle)
                        EEGData_metadata_3D = np.vstack([EEGData_metadata_3D, participant_EEGData_meta])

                        participant_EEGData = EEGData[participant_index,:,:]
                        EEGData = np.vstack([EEGData, participant_EEGData])

                        if pi % num_participant == 0 and pi > 0:
                            participant_cycle += 1

                #Encode data
                if autoencoder_filename != None:
                    EEGData = encode_data(autoencoder_filename, EEGData)
                
                #Average data per participant and condition
                EEGData = averageEEG(EEGData_metadata_3D[:,0], EEGData)

                #Extract outcome and feature data
                Y_train = EEGData[:,0,0]
                
                if features: #If extracting features
                    X_train = np.array(extractFeatures(EEGData[:,1:,:])) #Extract features
                    X_train = np.array([train_sample.T.flatten() for train_sample in X_train])
                    X_train = scale(X_train, axis=0) #Scale data within each trial
                else:
                    X_train = np.array([train_sample.T.flatten() for train_sample in EEGData[:,1:,:]])
                    X_train = scale(X_train, axis=1) #Scale across timeseries within trials
                
                #Create augmented dataset
                if addSyntheticData==1: #If this is augmented analyses
                    Y_train = np.concatenate((Y_train,syn_Y_train)) #Combine empirical and synthetic outcomes
                    X_train = np.concatenate((X_train,syn_X_train)) #Combine empirical and synthetic features
                    
                #Shuffle order of samples
                trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
                X_train = X_train[trainShuffle,:]
                Y_train = Y_train[trainShuffle]

                #Report current analyses
                print(classifier)
                if addSyntheticData == 1: 
                    print('Augmented')
                elif addSyntheticData == 0:
                    print('Empirical')
                else:
                    print('Oversampling')
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
                if addSyntheticData==1:
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