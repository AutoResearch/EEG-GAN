###############################################
## LOAD MODULES                              ##
###############################################
import os
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

def main(multiprocessing, features, validationOrTest, dataSampleSizes, syntheticDataOptions, classifiers, num_series, component='N400', prop_synthetic=None):
    
    ###############################################
    ## SETUP                                     ##
    ###############################################

    #Determine multiprocessing
    if multiprocessing:
        #Initiate multiprocessing manager
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count() + 2)

        #Initiate writer
        writer = pool.apply_async(write_classification, (q,))

    #Make Classification Results directory if it doesn't exist
    if not os.path.exists(f'classification/ERPCORE/{component}/Classification Results'):
        os.makedirs(f'classification/ERPCORE/{component}/Classification Results')

    ###############################################
    ## REPORT                                    ##
    ###############################################
        
    #Display parameters
    print('data Sample Sizes: ' )
    print(dataSampleSizes)
    print('Classification Data: ' + validationOrTest)
    print('Classifiers: ' + str(classifiers))

    if features:
        print('Features: Extracted Features')

    ###############################################
    ## CLASSIFICATION                            ##
    ###############################################
    jobs = []
    for series in range(num_series):
        for electrode_number in electrode_numbers: #Iterate through the electrodes
            y_test, x_test = load_test_data(validationOrTest, electrode_number, features, component=component)

            for classifier in classifiers: #Iterate through classifiers
                for addSyntheticData in syntheticDataOptions: #Iterate through analyses
                    for dataSampleSize in dataSampleSizes: #Iterate through sample sizes
                        for run in range(5): #Conduct analyses 5 times per sample size
                            if multiprocessing:
                                job = pool.apply_async(run_classification, args=(q,
                                                                                multiprocessing,
                                                                                validationOrTest, 
                                                                                features, 
                                                                                electrode_number, 
                                                                                classifier, 
                                                                                addSyntheticData, 
                                                                                dataSampleSize,  
                                                                                series,
                                                                                run,
                                                                                y_test, 
                                                                                x_test,
                                                                                component,
                                                                                prop_synthetic))
                                jobs.append(job)
                            else:
                                currentFilename, toWrite = run_classification(None,
                                                                              multiprocessing,
                                                                              validationOrTest, 
                                                                              features, 
                                                                              electrode_number, 
                                                                              classifier, 
                                                                              addSyntheticData, 
                                                                              dataSampleSize,  
                                                                              series,
                                                                              run,
                                                                              y_test, 
                                                                              x_test,
                                                                              component,
                                                                              prop_synthetic)
                                
                                write_classification(None, multiprocessing, currentFilename, toWrite)
        
    if multiprocessing:
        for job in jobs:
            job.get()

        q.put(['kill',''])
        pool.close()
        pool.join()

def run_classification(q, multiprocessing, validationOrTest, features, electrode_number, classifier, addSyntheticData, dataSampleSize, series, run, y_test, x_test, component='N400', prop_synthetic=None):

    print(f'ANALYSIS STARTED: Series = {series}, Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')

    #Base save file names
    generic_filename = f'classification/ERPCORE/{component}/Classification Results/XXANALYSISXXPredictions_e{electrode_number}_XXCLASSXX.csv'

    #Add features tag if applied
    if features:
        generic_filename = generic_filename.split('.csv')[0]+'_Features.csv'

    #Add test tag if test set being used
    if validationOrTest == 'test':
        generic_filename = generic_filename.split('.csv')[0]+'_TestClassification.csv'

    #Determine current filenames
    generic_filename = generic_filename.replace('XXCLASSXX', classifier).replace('XXANALYSISXX', addSyntheticData)

    ###############################################
    ## SYNTHETIC PROCESSING                      ##
    ###############################################
    if addSyntheticData == 'gan':
        synFilename_0 = f"generated_samples/ERPCORE/{component}/Training Datasets/gan_erpcore_{component}_SS{dataSampleSize}_Run0{run}_c0.csv"
        synFilename_1 = f"generated_samples/ERPCORE/{component}/Training Datasets/gan_erpcore_{component}_SS{dataSampleSize}_Run0{run}_c1.csv"
        syn_Y_train, syn_X_train = load_synthetic(synFilename_0, synFilename_1, features, prop_synthetic)

    elif addSyntheticData == 'vae':
        synFilename_0 = f"generated_samples/ERPCORE/{component}/Training Datasets/vae_erpcore_{component}_SS{dataSampleSize}_Run0{run}_c0.csv"
        synFilename_1 = f"generated_samples/ERPCORE/{component}/Training Datasets/vae_erpcore_{component}_SS{dataSampleSize}_Run0{run}_c1.csv"
        syn_Y_train, syn_X_train = load_synthetic(synFilename_0, synFilename_1, features, prop_synthetic)

    ###############################################
    ## EMPIRICAL PROCESSING                      ##
    ###############################################s
    
    #Load empirical data
    tempFilename = f'data/ERPCORE/{component}/Training Datasets/erpcore_{component}_SS{dataSampleSize}_Run0{run}.csv'
    EEGData_metadata = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)[:,:4]
    EEGData_metadata_3D = EEGData_metadata[EEGData_metadata[:,3] == np.unique(EEGData_metadata[:,3])[0],:]
    EEGData_dataloader = Dataloader(tempFilename, kw_conditions='Condition', kw_channel='Electrode')
    EEGData = EEGData_dataloader.get_data(shuffle=False).detach().numpy()

    #Oversampling analysis
    #TODO: Make this section a function
    if addSyntheticData == 'over':
        num_participant = np.unique(EEGData_metadata[:,0]).shape[0]
        participant_IDs = np.tile(np.unique(EEGData_metadata[:,0]),50)[:50]
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

    if addSyntheticData == 'gaus' or addSyntheticData == 'rev' or addSyntheticData == 'neg' or addSyntheticData == 'smooth':
        for sample_idx in range(EEGData.shape[0]):
            x_ = torch.as_tensor(EEGData[[sample_idx],1:,:])
            if np.random.rand() < .5: #Only half are transformed
                for e in range(x_.shape[-1]):
                    x_sample = x_[0,:,e]
                    if addSyntheticData == 'gaus':
                        X_tr = x_sample + np.random.normal(0, .1, 128)
                    elif addSyntheticData == 'rev':
                        X_tr = torch.flip(x_sample, (0,)) 
                    elif addSyntheticData == 'neg':
                        X_tr = -x_sample
                    elif addSyntheticData == 'smooth':

                        #Determine how much and where the data will be removed
                        mask_len_samples = int((np.random.uniform()*128)/6.67)+10 # 10-25% of the data
                        start_location = np.random.choice(range(0,128-mask_len_samples))

                        #Taken from braindecode
                        batch_size = 1 #Currently just one sample
                        n_channels = 1 #One electrode at a time
                        seq_len = x_sample.shape[0]
                        t = torch.arange(seq_len).float()
                        t = t.repeat(batch_size, n_channels, 1)
                        mask_start_per_sample = torch.tensor([start_location]).reshape(1,1,-1) #TODO: Randomize the 10
                        s = 1000 / seq_len
                        mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
                                torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))).float()[0,0,:]
                        X_tr = x_sample * mask

                    else:
                        X_tr = x_sample
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
            optimal_params, predictScore = kNearestNeighbor(X_train, Y_train, x_test, y_test)
        else:
            print('Unknown classifier')
            optimal_params = []
            predictScore = []

        print(f'ANALYSIS COMPLETE: Series = {series}, Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')
                    
        ###############################################
        ## SAVE DATA                                 ##
        ###############################################
            
        #Create list of what to write
        toWrite = [str(dataSampleSize),str(run),'0',str(predictScore),str(time.time()-startTime),optimal_params.best_params_]

        #Write data to file
        currentFilename = generic_filename

        if multiprocessing:
            q.put([currentFilename, toWrite])
    
    except:
        currentFilename = 'FAILED'
        toWrite = ''
        if multiprocessing:
            q.put(['FAILED', ''])
        print(f'ANALYSIS FAILED: Analysis = {addSyntheticData}, Classifier = {classifier}, Electrode = {electrode_number}, Sample Size = {dataSampleSize}, Run = {run}')

    return currentFilename, toWrite

#def write_classification(q, currentFilename, toWrite):
def write_classification(q, multiprocessing=True, currentFilename=None, toWrite=None):


    while True:
        if multiprocessing:
            #Receive data from classification function
            currentFilename, toWrite = q.get()

            if currentFilename == 'kill':
                print('All classifications complete.')
                break 

        if currentFilename != 'FAILED':
            with open(currentFilename, 'a') as f:
                for currentWrite in toWrite: #Iterate through write list
                    f.write(str(currentWrite)) #Write current item
                    if not currentWrite==toWrite[-1]: #If not the last item
                        f.write(',') #Add comma
                f.write('\n') #Creates new line
                f.flush() #Clears the internal buffer

        if not multiprocessing:
            break

if __name__ == '__main__':

    ###############################################
    ## USER INPUTS                               ##
    ###############################################
    
    #Determine inputs
    features = False #Datatype: False = Full Data, True = Features data
    validationOrTest = 'test' #'validation' or 'test' set to predict
    dataSampleSizes = ['005', '010', '015', '020'] #Which sample sizes to include
    syntheticDataOptions = ['emp','gan','vae','over','gaus','rev','neg','smooth'] #['emp','gan','vae','over','gaus','rev','neg','smooth'] #The code will iterate through this list. emp = empirical classifications, gan = gan-augmented classifications, vae = vae-augmented classification, over = oversampling classification
    classifiers = ['SVM','RF','KNN','NN','LR'] #The code will iterate through this list #NOTE NN AND LR USE THEIR OWN MULTIPROCESSING AND SLOWS THINGS WHEN RUN WITH MP, SO SHOULD BE RUN ONLY ALONE OR TOGETHER
    electrode_numbers = [1] #Which electrode to predict
    num_series = 10 #Number of times to run all classifications
    component = 'N170' #Which component of interest
    prop_synthetic = 1 #Proportion of synthetic participants to generate (1 = SS, 2 = 2*SS, etc.)

    #Split classifiers to multiprocessing vs not
    mp_classifiers = [c for c in classifiers if c != 'NN' and c != 'LR']
    nmp_classifiers = [c for c in classifiers if c == 'NN' or c == 'LR']

    ###############################################
    ## RUN CLASSIFICATION                        ##
    ###############################################

    if len(nmp_classifiers) > 0:
        print(f'Some classifiers you chose are not compatible with multiprocessing. \
              The classifers will be split into multiprocessing compatible versus not and run sequentially. \
              First we will run classifiers {mp_classifiers} using mutliprocessing, \
              then we will run classifiers {nmp_classifiers} without multiprocessing.')

    for i, current_classifiers in enumerate([mp_classifiers, nmp_classifiers]):
        if current_classifiers: 
            multiprocessing = True if i == 0 else False
            main(multiprocessing, features, validationOrTest, dataSampleSizes, syntheticDataOptions, current_classifiers, num_series, component, prop_synthetic)

    '''

    Analyses:
    emp: empirical
    gan: GAN-augmented
    vae: VAE-Augmented

    over: Oversampling
    gaus: Guassian Noise augmentation
    rev: Time Reverse
    neg: Polarity reverse
    smooth: Removed portions of data

    Classifiers:
    NN: Vanilla Neural Network
    SVM: Support Vector Machines
    LR: Logistic Regression
    RF: Random Forest
    KNN: K-Nearest Neighbours

    '''

 