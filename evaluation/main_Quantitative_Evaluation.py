import numpy as np
from scipy import signal
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

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



#############################################################
## LOAD DATA ##
#############################################################

REWP_IDs_c0, REWP_IDs_c1, REWP_eeg_c0, REWP_eeg_c1, REWP_gan_c0, REWP_gan_c1, REWP_vae_c0, REWP_vae_c1 = load_data(f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_len100.csv', 
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/gan_ep2000_p500_e1_full.csv',
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/vae_p500_e1_full.csv')

REWP8_IDs_c0, REWP8_IDs_c1,REWP8_eeg_c0, REWP8_eeg_c1, REWP8_gan_c0, REWP8_gan_c1, REWP8_vae_c0, REWP8_vae_c1 = load_data(f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e8_len100.csv', 
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/gan_ep2000_p500_e8_full.csv',
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/vae_p500_e8_full.csv',
                                                                                            select_electrode=7)
    
N2P3_IDs_c0, N2P3_IDs_c1, N2P3_eeg_c0, N2P3_eeg_c1, N2P3_gan_c0, N2P3_gan_c1, N2P3_vae_c0, N2P3_vae_c1 = load_data(f'data/Antisaccade/Training Datasets/antisaccade_left_full_cleaned.csv', 
                                                                                            f'generated_samples/Antisaccade/Training Datasets/gan_antisaccade_full_cleaned.csv',
                                                                                            f'generated_samples/Antisaccade/Training Datasets/vae_antisaccade_full_cleaned.csv')

N170_IDs_c0, N170_IDs_c1, N170_eeg_c0, N170_eeg_c1, N170_gan_c0, N170_gan_c1, N170_vae_c0, N170_vae_c1 = load_data(f'data/ERPCORE/N170/Training Datasets/erpcore_N170_full_cleaned.csv', 
                                                                                            f'generated_samples/ERPCORE/N170/Training Datasets/gan_erpcore_N170_full_cleaned.csv',
                                                                                            f'generated_samples/ERPCORE/N170/Training Datasets/vae_erpcore_N170_full_cleaned.csv')

N2PC_IDs_c0, N2PC_IDs_c1, N2PC_eeg_c0, N2PC_eeg_c1, N2PC_gan_c0, N2PC_gan_c1, N2PC_vae_c0, N2PC_vae_c1 = load_data(f'data/ERPCORE/N2PC/Training Datasets/erpcore_N2PC_full_cleaned.csv', 
                                                                                            f'generated_samples/ERPCORE/N2PC/Training Datasets/gan_erpcore_N2PC_full_cleaned.csv',
                                                                                            f'generated_samples/ERPCORE/N2PC/Training Datasets/vae_erpcore_N2PC_full_cleaned.csv')

#############################################################
## NEURAL NETWORK ##
#############################################################

#Load empirical data
tempFilename = f'data/ERPCORE/{component}/Training Datasets/erpcore_{component}_SS{dataSampleSize}_Run0{run}.csv'
EEGData_metadata = np.genfromtxt(tempFilename, delimiter=',', skip_header=1)[:,:4]
EEGData_metadata_3D = EEGData_metadata[EEGData_metadata[:,3] == np.unique(EEGData_metadata[:,3])[0],:]
EEGData_dataloader = Dataloader(tempFilename, kw_conditions='Condition', kw_channel='Electrode')
EEGData = EEGData_dataloader.get_data(shuffle=False).detach().numpy()


#Average data per participant and condition
EEGData = averageEEG(EEGData_metadata_3D[:,0], EEGData)

#Extract outcome and feature data
Y_train = EEGData[:,0,0]

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

optimal_params, predictScore = neuralNetwork(X_train, Y_train, x_test, y_test)