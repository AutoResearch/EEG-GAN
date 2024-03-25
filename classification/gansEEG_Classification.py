###############################################
## LOAD MODULES                              ##
###############################################
import numpy as np
from sklearn.preprocessing import scale
import random as rnd
import time
import multiprocessing as mp

from helpers.dataloader import Dataloader
from classification_functions import *

###############################################
## DEFINE FUNCTIONS                          ##
###############################################

def main():
    
    ###############################################
    ## SETUP                                     ##
    ###############################################

    #Initiate multiprocessing manager
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    #Initiate writer
    writer = pool.apply_async(write_classification, (q,))

    ###############################################
    ## USER INPUTS                               ##
    ###############################################
    features = False #Datatype: False = Full Data, True = Features data
    validationOrTest = 'validation' #'validation' or 'test' set to predict
    dataSampleSizes = ['005', '010', '015', '020', '030', '060', '100'] #Which sample sizes to include
    syntheticDataOptions = ['gaus']#['emp', 'gan', 'vae'] #The code will iterate through this list. emp = empirical classifications, gan = gan-augmented classifications, vae = vae-augmented classification, over = oversampling classification
    classifiers = ['NN', 'SVM', 'LR', 'RF', 'KNN'] #The code will iterate through this list
    electrode_numbers = [1]

    '''

    Analyses:
    emp: empirical
    gan: GAN-augmented
    vae: VAE-Augmented

    gaus: Guassian Noise augmentation

    Classifiers:
    NN: Vanilla Neural Network
    SVM: Support Vector Machines
    LR: Logistic Regression
    RF: Random Forest
    KNN: K-Nearest Neighbours

    '''

    ###############################################
    ## SETUP                                     ##
    ###############################################
        
    #Display parameters
    print('data Sample Sizes: ' )
    print(dataSampleSizes)
    print('Classification Data: ' + validationOrTest)

    if features:
        print('Features: Extracted Features')

    ###############################################
    ## CLASSIFICATION                            ##
    ###############################################
    jobs = []
    for electrode_number in electrode_numbers: #Iterate through the electrodes
        y_test, x_test = load_test_data(validationOrTest, electrode_number, features)

        for classifier in classifiers: #Iterate through classifiers
            for addSyntheticData in syntheticDataOptions: #Iterate through analyses
                for dataSampleSize in dataSampleSizes: #Iterate through sample sizes
                    for run in range(5): #Conduct analyses 5 times per sample size
                        job = pool.apply_async(run_classification, args=(q,
                                                                        validationOrTest, 
                                                                        features, 
                                                                        electrode_number, 
                                                                        classifier, 
                                                                        addSyntheticData, 
                                                                        dataSampleSize, 
                                                                        run, 
                                                                        y_test, 
                                                                        x_test))
                        jobs.append(job)
        
    for job in jobs:
        job.get()

    q.put(['kill',''])
    pool.close()
    pool.join()

def run_classification(q, validationOrTest, features, electrode_number, classifier, addSyntheticData, dataSampleSize, run, y_test, x_test):

    print(f'ANALYSIS STARTED: Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')

    #Base save file names
    augFilename = f'classification/Classification Results/augmentedPredictions_e{electrode_number}_XX.csv'
    empFilename = f'classification/Classification Results/empiricalPredictions_e{electrode_number}_XX.csv'
    ovsFilename = f'classification/Classification Results/oversamplingPredictions_e{electrode_number}_XX.csv'
    vaeFilename = f'classification/Classification Results/vaePredictions_e{electrode_number}_XX.csv'
    gausFilename = f'classification/Classification Results/gaussianPredictions_e{electrode_number}_XX.csv'

    #Add features tag if applied
    if features:
        augFilename = augFilename.split('.csv')[0]+'_Features.csv'
        empFilename = empFilename.split('.csv')[0]+'_Features.csv'
        ovsFilename = ovsFilename.split('.csv')[0]+'_Features.csv'
        vaeFilename = vaeFilename.split('.csv')[0]+'_Features.csv'
        gausFilename = gausFilename.split('.csv')[0]+'_Features.csv'

    #Add test tag if test set being used
    if validationOrTest == 'test':
        augFilename = augFilename.split('.csv')[0]+'_TestClassification.csv'
        empFilename = empFilename.split('.csv')[0]+'_TestClassification.csv'
        ovsFilename = ovsFilename.split('.csv')[0]+'_TestClassification.csv'
        vaeFilename = vaeFilename.split('.csv')[0]+'_TestClassification.csv'
        gausFilename = gausFilename.split('.csv')[0]+'_TestClassification.csv'

    #Determine current filenames
    currentAugFilename = augFilename.replace('XX',classifier)
    currentEmpFilename = empFilename.replace('XX',classifier)
    currentOvsFilename = ovsFilename.replace('XX',classifier)
    currentVaeFilename = vaeFilename.replace('XX',classifier)
    currentGausFilename = gausFilename.replace('XX',classifier)

    ###############################################
    ## SYNTHETIC PROCESSING                      ##
    ###############################################
    if addSyntheticData == 'gan':
        synFilename_0 = f"generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c0.csv"
        synFilename_1 = f"generated_samples/Reinforcement Learning/aegan_ep2000_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c1.csv"
        syn_Y_train, syn_X_train = load_synthetic(synFilename_0, synFilename_1, features)

    elif addSyntheticData == 'vae':
        synFilename_0 = f"generated_samples/Reinforcement Learning/vae_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c0.csv"
        synFilename_1 = f"generated_samples/Reinforcement Learning/vae_e{electrode_number}_SS{dataSampleSize}_Run0{run}_c1.csv"
        syn_Y_train, syn_X_train = load_synthetic(synFilename_0, synFilename_1, features)

    ###############################################
    ## EMPIRICAL PROCESSING                      ##
    ###############################################s
    
    #Load empirical data
    tempFilename = f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e{electrode_number}_SS{dataSampleSize}_Run0{run}.csv'
    EEGData_metadata = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)[:,:4]
    EEGData_metadata_3D = EEGData_metadata[EEGData_metadata[:,3] == np.unique(EEGData_metadata[:,3])[0],:]
    EEGData_dataloader = Dataloader(tempFilename, col_label='Condition', channel_label='Electrode')
    EEGData = EEGData_dataloader.get_data(shuffle=False).detach().numpy()

    #Oversampling analysis
    if addSyntheticData == 'over':
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

    if addSyntheticData == 'gaus':
        for sample_idx in range(EEGData.shape[0]):
            x_ = torch.as_tensor(EEGData[[sample_idx],1:,:])
            if np.random.rand() < .5: #Only half are transformed
                for e in range(x_.shape[-1]):
                    X_tr = x_[0,:,e] + np.random.normal(0, .1, 100) #TODO: This is a diff augmentation (Mean = 0, STD = .1, size = 100)
                    EEGData[sample_idx,1:,e] = X_tr
    
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
    if addSyntheticData == 'gan' or addSyntheticData == 'vae': #If this is augmented analyses
        Y_train = np.concatenate((Y_train,syn_Y_train)) #Combine empirical and synthetic outcomes
        X_train = np.concatenate((X_train,syn_X_train)) #Combine empirical and synthetic features
        
    #Shuffle order of samples
    trainShuffle = rnd.sample(range(len(X_train)),len(X_train))
    X_train = X_train[trainShuffle,:]
    Y_train = Y_train[trainShuffle]

    ###############################################
    ## CLASSIFIER                                ##
    ###############################################
    
    #Begin timer
    startTime = time.time()

    try:
        #Run classifier
        if classifier == 'NN':
            optimal_params, predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)
        elif classifier == 'SVM':
            optimal_params, predictScore = supportVectorMachine(X_train, Y_train, x_test, y_test)
        elif classifier == 'LR':
            optimal_params, predictScore = logisticRegression(X_train, Y_train, x_test, y_test)
        elif classifier == 'RF':
            optimal_params, predictScore = randomForest(X_train, Y_train, x_test, y_test)
        elif classifier == 'KNN':
            optimal_params, predictScore = kNearestNeighbor(X_train, Y_train, x_test, y_test, X_train.shape[0])
        else:
            print('Unknown classifier')
            optimal_params = []
            predictScore = []

        print(f'ANALYSIS COMPLETE: Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')
                    
        ###############################################
        ## SAVE DATA                                 ##
        ###############################################
            
        #Create list of what to write
        toWrite = [str(dataSampleSize),str(run),'0',str(predictScore),str(time.time()-startTime),optimal_params.best_params_]

        #Write data to file
        if addSyntheticData=='gan':
            currentFilename = currentAugFilename
        elif addSyntheticData=='emp':
            currentFilename = currentEmpFilename
        elif addSyntheticData=='over':
            currentFilename = currentOvsFilename
        elif addSyntheticData=='vae':
            currentFilename = currentVaeFilename
        elif addSyntheticData=='gaus':
            currentFilename = currentGausFilename
        else:
            raise NotImplementedError('Analysis not recognized.')
        
        q.put([currentFilename, toWrite])
    
    except:
        currentFilename = 'FAILED'
        toWrite = ''
        q.put(['FAILED', ''])
        print(f'ANALYSIS FAILED: Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')

    return currentFilename, toWrite

#def write_classification(q, currentFilename, toWrite):
def write_classification(q):

    while True:
        #Receive data from classification function
        currentFilename, toWrite = q.get()
        
        if currentFilename == 'kill':
            print('killed')
            break 

        if currentFilename != 'FAILED':
            with open(currentFilename, 'a') as f:
                for currentWrite in toWrite: #Iterate through write list
                    f.write(str(currentWrite)) #Write current item
                    if not currentWrite==toWrite[-1]: #If not the last item
                        f.write(',') #Add comma
                f.write('\n') #Creates new line
                f.flush() #Clears the internal buffer

if __name__ == '__main__':
    main()