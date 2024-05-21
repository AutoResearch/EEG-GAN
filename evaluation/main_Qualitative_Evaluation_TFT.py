###############################################
## LOAD MODULES                              ##
###############################################
import os
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt

###############################################
## FUNCTIONS                                 ##
###############################################

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

def constrain(data, num_bins=10):
    #transform data that is between 0 and 1 to be between -1 and 1
    data = (data*2)-1

    #Determining bin edges
    bin_edges = np.linspace(-1, 1, num_bins+1)

    #Constraining data to be in bins
    for i in range(num_bins):
        if bin_edges[i] < 0:
            data[(data >= bin_edges[i]) & (data < bin_edges[i+1])] = bin_edges[i]
        elif bin_edges[i] == 0 or bin_edges[i+1] == 0:
            data[(data >= bin_edges[i]) & (data < bin_edges[i+1])] = 0
        else:
            data[(data > bin_edges[i]) & (data <= bin_edges[i+1])] = bin_edges[i+1]

    return data

def frequency_transform(data):
    
    #Create a new array to store the transformed data
    transformedData = np.zeros((data.shape[0], data.shape[1]//2+1))
    
    #Iterate through the data
    for i, dat in enumerate(data):
        transformedData[i,:] = np.abs(scipy.fft.rfft(dat))[:(data.shape[1]//2)+1]
    
    return transformedData

def time_frequency_wavelets(sample, sampling_period, wavelet = 'cmor1.5-1.0'):
    widths = np.geomspace(1, 1024, num=100)
    
    tf, freqs = pywt.cwt(sample, widths, wavelet, sampling_period=sampling_period)
    tf = np.abs(tf[:-1, :-1]) #Take the absolute value of the wavelet transform

    return freqs, tf

def time_frequency_transform(data, speriod=1/1000, label=''):

    tfrs = np.empty((data.shape[0], 99, data.shape[1]-1)) #Create empty array to store output

    loop =  tqdm(enumerate(data))
    loop.total = data.shape[0]
    for index, sample in loop:
        
        frex, tfr = time_frequency_wavelets(sample, speriod)
        tfrs[index, :, :] = tfr
        loop.update(1)

    return frex, tfrs

###############################################
## LOAD AND PROCESS DATA                     ##
###############################################
def main(try_=None, load_data=True):
    
    ## EMPIRICAL ##
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
    
    if not load_data: 
        REWP_eeg_c0, REWP_eeg_c1, REWP_gan_c0, REWP_gan_c1, REWP_vae_c0, REWP_vae_c1 = load_data(f'data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e1_len100.csv', 
                                                                                                f'generated_samples/Reinforcement Learning/Full Datasets/gan_ep2000_p500_e1_full.csv',
                                                                                                f'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e1_full.csv')

        REWP8_eeg_c0, REWP8_eeg_c1, REWP8_gan_c0, REWP8_gan_c1, REWP8_vae_c0, REWP8_vae_c1 = load_data(f'data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e8_len100.csv', 
                                                                                                f'generated_samples/Reinforcement Learning/Full Datasets/gan_ep2000_p500_e8_full.csv',
                                                                                                f'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e8_full.csv',
                                                                                                select_electrode=7)
        
        N2P3_eeg_c0, N2P3_eeg_c1, N2P3_gan_c0, N2P3_gan_c1, N2P3_vae_c0, N2P3_vae_c1 = load_data(f'data/Antisaccade/Full Datasets/antisaccade_left_full_cleaned.csv', 
                                                                                                f'generated_samples/Antisaccade/Full Datasets/gan_antisaccade_full_cleaned.csv',
                                                                                                f'generated_samples/Antisaccade/Full Datasets/vae_antisaccade_full_cleaned.csv')

        N170_eeg_c0, N170_eeg_c1, N170_gan_c0, N170_gan_c1, N170_vae_c0, N170_vae_c1 = load_data(f'data/ERPCORE/N170/Full Datasets/erpcore_N170_full_cleaned.csv', 
                                                                                                f'generated_samples/ERPCORE/N170/Full Datasets/gan_erpcore_N170_full_cleaned.csv',
                                                                                                f'generated_samples/ERPCORE/N170/Full Datasets/vae_erpcore_N170_full_cleaned.csv')

        N2PC_eeg_c0, N2PC_eeg_c1, N2PC_gan_c0, N2PC_gan_c1, N2PC_vae_c0, N2PC_vae_c1 = load_data(f'data/ERPCORE/N2PC/Full Datasets/erpcore_N2PC_full_cleaned.csv', 
                                                                                                f'generated_samples/ERPCORE/N2PC/Full Datasets/gan_erpcore_N2PC_full_cleaned.csv',
                                                                                                f'generated_samples/ERPCORE/N2PC/Full Datasets/vae_erpcore_N2PC_full_cleaned.csv')

        frexREWP, REWP_tft_c0 = time_frequency_transform(REWP_eeg_c0, speriod=1/100)
        _, REWP_tft_c1 = time_frequency_transform(REWP_eeg_c1, speriod=1/100)
        _, REWP8_tft_c0 = time_frequency_transform(REWP8_eeg_c0, speriod=1/100)
        _, REWP8_tft_c1 = time_frequency_transform(REWP8_eeg_c1, speriod=1/100)
        frexN2P3, N2P3_tft_c0 = time_frequency_transform(N2P3_eeg_c0, speriod=1/125)
        _, N2P3_tft_c1 = time_frequency_transform(N2P3_eeg_c1, speriod=1/125)
        frexN170, N170_tft_c0 = time_frequency_transform(N170_eeg_c0, speriod=1/128)
        _, N170_tft_c1 = time_frequency_transform(N170_eeg_c1, speriod=1/128)
        frexN2PC, N2PC_tft_c0 = time_frequency_transform(N2PC_eeg_c0, speriod=1/128)
        _, N2PC_tft_c1 = time_frequency_transform(N2PC_eeg_c1, speriod=1/128)

        _, gan_REWP_tft_c0 = time_frequency_transform(REWP_gan_c0, speriod=1/100)
        _, gan_REWP_tft_c1 = time_frequency_transform(REWP_gan_c1, speriod=1/100)
        _, gan_REWP8_tft_c0 = time_frequency_transform(REWP8_gan_c0, speriod=1/100)
        _, gan_REWP8_tft_c1 = time_frequency_transform(REWP8_gan_c1, speriod=1/100)
        _, gan_N2P3_tft_c0 = time_frequency_transform(N2P3_gan_c0, speriod=1/125)
        _, gan_N2P3_tft_c1 = time_frequency_transform(N2P3_gan_c1, speriod=1/125)
        _, gan_N170_tft_c0 = time_frequency_transform(N170_gan_c0, speriod=1/128)
        _, gan_N170_tft_c1 = time_frequency_transform(N170_gan_c1, speriod=1/128)
        _, gan_N2PC_tft_c0 = time_frequency_transform(N2PC_gan_c0, speriod=1/128)
        _, gan_N2PC_tft_c1 = time_frequency_transform(N2PC_gan_c1, speriod=1/128)

        _, vae_REWP_tft_c0 = time_frequency_transform(REWP_vae_c0, speriod=1/100)
        _, vae_REWP_tft_c1 = time_frequency_transform(REWP_vae_c1, speriod=1/100)
        _, vae_REWP8_tft_c0 = time_frequency_transform(REWP8_vae_c0, speriod=1/100)
        _, vae_REWP8_tft_c1 = time_frequency_transform(REWP8_vae_c1, speriod=1/100)
        _, vae_N2P3_tft_c0 = time_frequency_transform(N2P3_vae_c0, speriod=1/125)
        _, vae_N2P3_tft_c1 = time_frequency_transform(N2P3_vae_c1, speriod=1/125)
        _, vae_N170_tft_c0 = time_frequency_transform(N170_vae_c0, speriod=1/128)
        _, vae_N170_tft_c1 = time_frequency_transform(N170_vae_c1, speriod=1/128)
        _, vae_N2PC_tft_c0 = time_frequency_transform(N2PC_vae_c0, speriod=1/128)
        _, vae_N2PC_tft_c1 = time_frequency_transform(N2PC_vae_c1, speriod=1/128)

        #Write all data to a file
        np.savez('data/qualitative_evaluation.npz', frexREWP=frexREWP, frexN2P3=frexN2P3, frexN170=frexN170, frexN2PC=frexN2PC, REWP_tft_c0=REWP_tft_c0, REWP_tft_c1=REWP_tft_c1, REWP8_tft_c0=REWP8_tft_c0, REWP8_tft_c1=REWP8_tft_c1, N2P3_tft_c0=N2P3_tft_c0, N2P3_tft_c1=N2P3_tft_c1, N170_tft_c0=N170_tft_c0, N170_tft_c1=N170_tft_c1, N2PC_tft_c0=N2PC_tft_c0, N2PC_tft_c1=N2PC_tft_c1, gan_REWP_tft_c0=gan_REWP_tft_c0, gan_REWP_tft_c1=gan_REWP_tft_c1, gan_REWP8_tft_c0=gan_REWP8_tft_c0, gan_REWP8_tft_c1=gan_REWP8_tft_c1, gan_N2P3_tft_c0=gan_N2P3_tft_c0, gan_N2P3_tft_c1=gan_N2P3_tft_c1, gan_N170_tft_c0=gan_N170_tft_c0, gan_N170_tft_c1=gan_N170_tft_c1, gan_N2PC_tft_c0=gan_N2PC_tft_c0, gan_N2PC_tft_c1=gan_N2PC_tft_c1, vae_REWP_tft_c0=vae_REWP_tft_c0, vae_REWP_tft_c1=vae_REWP_tft_c1, vae_REWP8_tft_c0=vae_REWP8_tft_c0, vae_REWP8_tft_c1=vae_REWP8_tft_c1, vae_N2P3_tft_c0=vae_N2P3_tft_c0, vae_N2P3_tft_c1=vae_N2P3_tft_c1, vae_N170_tft_c0=vae_N170_tft_c0, vae_N170_tft_c1=vae_N170_tft_c1, vae_N2PC_tft_c0=vae_N2PC_tft_c0, vae_N2PC_tft_c1=vae_N2PC_tft_c1)
    
    else:

        print('Loading data...')
        #Load data
        temp = np.load('data/qualitative_evaluation.npz')
        #unpack npz file
        frexREWP = temp['frexREWP']
        frexN2P3 = temp['frexN2P3']
        frexN170 = temp['frexN170']
        frexN2PC = temp['frexN2PC']
        REWP_tft_c0 = temp['REWP_tft_c0']
        REWP_tft_c1 = temp['REWP_tft_c1']
        REWP8_tft_c0 = temp['REWP8_tft_c0']
        REWP8_tft_c1 = temp['REWP8_tft_c1']
        N2P3_tft_c0 = temp['N2P3_tft_c0']
        N2P3_tft_c1 = temp['N2P3_tft_c1']
        N170_tft_c0 = temp['N170_tft_c0']
        N170_tft_c1 = temp['N170_tft_c1']
        N2PC_tft_c0 = temp['N2PC_tft_c0']
        N2PC_tft_c1 = temp['N2PC_tft_c1']
        gan_REWP_tft_c0 = temp['gan_REWP_tft_c0']
        gan_REWP_tft_c1 = temp['gan_REWP_tft_c1']
        gan_REWP8_tft_c0 = temp['gan_REWP8_tft_c0']
        gan_REWP8_tft_c1 = temp['gan_REWP8_tft_c1']
        gan_N2P3_tft_c0 = temp['gan_N2P3_tft_c0']
        gan_N2P3_tft_c1 = temp['gan_N2P3_tft_c1']
        gan_N170_tft_c0 = temp['gan_N170_tft_c0']
        gan_N170_tft_c1 = temp['gan_N170_tft_c1']
        gan_N2PC_tft_c0 = temp['gan_N2PC_tft_c0']
        gan_N2PC_tft_c1 = temp['gan_N2PC_tft_c1']
        vae_REWP_tft_c0 = temp['vae_REWP_tft_c0']
        vae_REWP_tft_c1 = temp['vae_REWP_tft_c1']
        vae_REWP8_tft_c0 = temp['vae_REWP8_tft_c0']
        vae_REWP8_tft_c1 = temp['vae_REWP8_tft_c1']
        vae_N2P3_tft_c0 = temp['vae_N2P3_tft_c0']
        vae_N2P3_tft_c1 = temp['vae_N2P3_tft_c1']
        vae_N170_tft_c0 = temp['vae_N170_tft_c0']
        vae_N170_tft_c1 = temp['vae_N170_tft_c1']
        vae_N2PC_tft_c0 = temp['vae_N2PC_tft_c0']
        vae_N2PC_tft_c1 = temp['vae_N2PC_tft_c1']
        print('Data loading complete!')

    #######################################
    ## FIGURE S1
    #######################################

    #Plotting Function
    def plot_TFT(c0, c1, num_item, ylim, frex, num_rows=5):

        #Setup
        ax1 = plt.subplot(num_rows, 3, num_item)

        #Plot
        time = np.linspace(0,100,c0.shape[-1]+1)/100
        print(c0.shape, c1.shape, time.shape, frex.shape)
        c = constrain(norm(np.mean(c0,0) - np.mean(c1,0)),num_bins=100)
        im = plt.pcolormesh(time, frex, c, cmap='coolwarm')

        #Title
        if num_item == 1:
            plt.text(0.5, 1.2, 'Empirical', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            #plt.title('Empirical', fontsize=16, fontweight='bold')
        elif num_item == 2:
            plt.text(0.5, 1.2, 'GAN-Synthetic', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            #plt.title('GAN-Synthetic')
        elif num_item == 3:
            plt.text(0.5, 1.2, 'VAE-Synthetic', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            #plt.title('VAE-Synthetic')

        #Row labels
        if num_item == 1:
            plt.text(-0.2, 1.1, 'Reinforcement\nLearning (E1: FCz)', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        if num_item == 4:
            plt.text(-0.2, 1.1, 'Reinforcement\nLearning (E8: POz)', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        elif num_item == 7:
            plt.text(-0.2, 1.1, 'Anti-Saccade', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        elif num_item == 10:
            plt.text(-0.2, 1.1, 'Face Perception', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        elif num_item == 13:
            plt.text(-0.2, 1.1, 'Visual Search', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        
        #Labels
        if num_item > 12:
            plt.xlabel('Time (ms)')
        if num_item == 1 or num_item == 4 or num_item == 7 or num_item == 10 or num_item == 13:
            plt.ylabel('Frequency (Hz)')

        #Format
        plt.ylim([0,20])
        plt.yticks(np.linspace(0,20,5), ['0', '5', '10', '15', '20'])
        if num_item < 7:
            plt.xticks(np.linspace(0,1,7), ['-200', '0', '200', '400', '600', '800', '1000'])
        else:
            plt.xticks(np.linspace(0,1,6), ['-200', '0', '200', '400', '600', '800'])
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)

    #Plot
    num_rows = 5
    fig = plt.figure(figsize=(12,num_rows*3))

    plot_TFT(REWP_tft_c0, REWP_tft_c1, 1, [], frexREWP)
    plot_TFT(gan_REWP_tft_c0, gan_REWP_tft_c1, 2, [], frexREWP)
    plot_TFT(vae_REWP_tft_c0, vae_REWP_tft_c1, 3, [], frexREWP)
    plot_TFT(REWP8_tft_c0, REWP8_tft_c1, 4, [], frexREWP)
    plot_TFT(gan_REWP8_tft_c0, gan_REWP8_tft_c1, 5, [], frexREWP)
    plot_TFT(vae_REWP8_tft_c0, vae_REWP8_tft_c1, 6, [], frexREWP)
    plot_TFT(N2P3_tft_c0, N2P3_tft_c1, 7, [], frexN2P3)
    plot_TFT(gan_N2P3_tft_c0, gan_N2P3_tft_c1, 8, [], frexN2P3)
    plot_TFT(vae_N2P3_tft_c0, vae_N2P3_tft_c1, 9, [], frexN2P3)
    plot_TFT(N170_tft_c0, N170_tft_c1, 10, [], frexN170)
    plot_TFT(gan_N170_tft_c0, gan_N170_tft_c1, 11, [], frexN170)
    plot_TFT(vae_N170_tft_c0, vae_N170_tft_c1, 12, [], frexN170)
    plot_TFT(N2PC_tft_c0, N2PC_tft_c1, 13, [], frexN2PC)
    plot_TFT(gan_N2PC_tft_c0, gan_N2PC_tft_c1, 14, [], frexN2PC)
    plot_TFT(vae_N2PC_tft_c0, vae_N2PC_tft_c1, 15, [], frexN2PC)

    #Save
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, num_rows*4)

    fig.savefig(f'figures/Figure S2 - gan_tft_Evaluations.png', dpi=600)

if __name__ == '__main__':
    load_data = True
    main(load_data=load_data)
