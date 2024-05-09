import numpy as np
from scipy import signal

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
    EEG_data = np.genfromtxt(data, delimiter=',', skip_header=1)[:,1:] #Removes Participant Column
    EEG_data = np.delete(EEG_data, 1, 1) #Delete Unused Column (Trial)

    #Select Electrode
    if select_electrode:
        EEG_data = EEG_data[np.r_[EEG_data[:,1]==select_electrode],:]

    #Split into conditions
    c1_EEG_data = EEG_data[np.r_[EEG_data[:,0]==1],2:]  
    c0_EEG_data = EEG_data[np.r_[EEG_data[:,0]==0],2:] 

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

    return c1_EEG_data, c0_EEG_data, scaledc1ganData, scaledc0ganData, scaledc1vaeData, scaledc0vaeData

REWP_eeg_c0, REWP_eeg_c1, REWP_gan_c0, REWP_gan_c1, REWP_vae_c0, REWP_vae_c1 = load_data(f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e1_len100.csv', 
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/gan_ep2000_p500_e1_full.csv',
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/vae_p500_e1_full.csv')

REWP8_eeg_c0, REWP8_eeg_c1, REWP8_gan_c0, REWP8_gan_c1, REWP8_vae_c0, REWP8_vae_c1 = load_data(f'data/Reinforcement Learning/Training Datasets/ganTrialElectrodeERP_p500_e8_len100.csv', 
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/gan_ep2000_p500_e8_full.csv',
                                                                                            f'generated_samples/Reinforcement Learning/Training Datasets/vae_p500_e8_full.csv',
                                                                                            select_electrode=7)
    
N2P3_eeg_c0, N2P3_eeg_c1, N2P3_gan_c0, N2P3_gan_c1, N2P3_vae_c0, N2P3_vae_c1 = load_data(f'data/Antisaccade/Training Datasets/antisaccade_left_full_cleaned.csv', 
                                                                                            f'generated_samples/Antisaccade/Training Datasets/gan_antisaccade_full_cleaned.csv',
                                                                                            f'generated_samples/Antisaccade/Training Datasets/vae_antisaccade_full_cleaned.csv')

N170_eeg_c0, N170_eeg_c1, N170_gan_c0, N170_gan_c1, N170_vae_c0, N170_vae_c1 = load_data(f'data/ERPCORE/N170/Training Datasets/erpcore_N170_full_cleaned.csv', 
                                                                                            f'generated_samples/ERPCORE/N170/Training Datasets/gan_erpcore_N170_full_cleaned.csv',
                                                                                            f'generated_samples/ERPCORE/N170/Training Datasets/vae_erpcore_N170_full_cleaned.csv')

N2PC_eeg_c0, N2PC_eeg_c1, N2PC_gan_c0, N2PC_gan_c1, N2PC_vae_c0, N2PC_vae_c1 = load_data(f'data/ERPCORE/N2PC/Training Datasets/erpcore_N2PC_full_cleaned.csv', 
                                                                                            f'generated_samples/ERPCORE/N2PC/Training Datasets/gan_erpcore_N2PC_full_cleaned.csv',
                                                                                            f'generated_samples/ERPCORE/N2PC/Training Datasets/vae_erpcore_N2PC_full_cleaned.csv')
