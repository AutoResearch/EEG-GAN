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
import multiprocessing as mp
import sklearn.manifold as sklm

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
def main(run_TFT=True, process_synthetic=True, run_gan=True, run_vae=True, component='N400'):

    print('Loading data...')
    
    ## EMPIRICAL ##
    def load_data(data, gan_data, vae_data, run_gan=True, run_vae=True, process_synthetic=True):
        #Load EEG Data (Participant, Condition, Trial, Electrode, Time1, ...)
        EEG_data = np.genfromtxt(data, delimiter=',', skip_header=1)[:,1:] #Removes Participant Column
        EEG_data = np.delete(EEG_data, 1, 1) #Delete Unused Column (Trial)

        #Split into conditions
        c1_EEG_data = EEG_data[np.r_[EEG_data[:,0]==1],2:]  
        c0_EEG_data = EEG_data[np.r_[EEG_data[:,0]==0],2:] 
        c1_EEG_data = np.mean(c1_EEG_data, axis=0)
        c0_EEG_data = np.mean(c0_EEG_data, axis=0)

        if run_gan:
            ## GAN ##
            
            #Load and Process Synthetic Data (Condition, Electrode, Time0, ...)
            gan_fn_0 = gan_data.replace('.csv', '_c0.csv')
            gan_fn_1 = gan_data.replace('.csv', '_c1.csv')
            ganData0 = np.genfromtxt(gan_fn_0, delimiter=',', skip_header=1)
            ganData1 = np.genfromtxt(gan_fn_1, delimiter=',', skip_header=1)
            ganData = np.vstack((ganData0, ganData1))
            ganData = np.delete(ganData, 1,1) #Remove the electrode column but we need to keep it

            #Process synthetic data
            fftTempganData = ganData
            if process_synthetic:
                tempganData = filterEEG(ganData[:,1:])
                tempganData = baselineCorrect(tempganData)
            else:
                tempganData = ganData[:,1:]
            
            #Create and populate new array for processed synthetic data
            processedganData = np.zeros((len(tempganData),129))
            processedganData[:,0] = ganData[:,0]
            processedganData[:,1:] = np.array(tempganData)

            #Split into conditions
            lossganData = processedganData[np.r_[processedganData[:,0]==1],1:]
            winganData = processedganData[np.r_[processedganData[:,0]==0],1:]

            #Average data
            avgLossganData = np.mean(lossganData, axis=0)
            avgWinganData = np.mean(winganData, axis=0)

            #Scale synthetic data 
            EEGDataScale = np.max(c1_EEG_data)-np.min(c1_EEG_data) 
            EEGDataOffset = np.min(c1_EEG_data)
            ganDataScale = np.max(avgLossganData)-np.min(avgLossganData)
            ganDataOffset = np.min(avgLossganData)

            avgLossganData = (((avgLossganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
            avgWinganData = (((avgWinganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

            scaledLossganData = (((lossganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
            scaledWinganData = (((winganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

        if run_vae:
            ## VAE ##
            
            vae_fn_0 = vae_data.replace('.csv', '_c0.csv')
            vae_fn_1 = vae_data.replace('.csv', '_c1.csv')
            vaeData0 = np.genfromtxt(vae_fn_0, delimiter=',', skip_header=1)
            vaeData1 = np.genfromtxt(vae_fn_1, delimiter=',', skip_header=1)
            vaeData = np.vstack((vaeData0, vaeData1))
            vaeData = np.delete(vaeData, 1,1) #Remove the electrode column but we need to keep it

            #Process synthetic data
            if process_synthetic:
                tempvaeData = filterEEG(vaeData[:,1:])
                tempvaeData = baselineCorrect(tempvaeData)
            else:
                tempvaeData = vaeData[:,1:]

            #Create and populate new array for processed synthetic data
            processedvaeData = np.zeros((len(tempvaeData),129))
            processedvaeData[:,0] = vaeData[:,0]
            processedvaeData[:,1:] = np.array(tempvaeData)

            #Split into conditions
            lossvaeData = processedvaeData[np.r_[processedvaeData[:,0]==1],1:]
            winvaeData = processedvaeData[np.r_[processedvaeData[:,0]==0],1:]

            #Average data
            avgLossvaeData = np.mean(lossvaeData, axis=0)
            avgWinvaeData = np.mean(winvaeData, axis=0)

            #Scale synthetic data 
            EEGDataScale = np.max(c1_EEG_data)-np.min(c1_EEG_data)
            EEGDataOffset = np.min(c1_EEG_data)
            vaeDataScale = np.max(avgLossvaeData)-np.min(avgLossvaeData)
            vaeDataOffset = np.min(avgLossvaeData)

            avgLossvaeData = (((avgLossvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
            avgWinvaeData = (((avgWinvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

            scaledLossvaeData = (((lossvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
            scaledWinvaeData = (((winvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

        print('Data loading complete!')
    
        return c1_EEG_data, c0_EEG_data, scaledWinganData, scaledLossganData, scaledWinvaeData, scaledLossvaeData
    
    REWP_eeg_c0, REWP_eeg_c1, REWP_gan_c0, REWP_gan_c1, REWP_vae_c0, REWP_vae_c1 = load_data(f'data/ERPCORE/REWP/Full Datasets/erpcore_REWP_full_cleaned.csv', 
                                                                                             f'generated_samples/ERPCORE/REWP/Full Datasets/gan_erpcore_REWP_full_cleaned.csv',
                                                                                             f'generated_samples/ERPCORE/REWP/Full Datasets/vae_erpcore_REWP_full.csv')
    
    N2P3_eeg_c0, N2P3_eeg_c1, N2P3_gan_c0, N2P3_gan_c1, N2P3_vae_c0, N2P3_vae_c1 = load_data(f'data/ERPCORE/N2P3/Full Datasets/erpcore_N2P3_full_cleaned.csv', 
                                                                                             f'generated_samples/ERPCORE/N2P3/Full Datasets/gan_erpcore_N2P3_full_cleaned.csv',
                                                                                             f'generated_samples/ERPCORE/N2P3/Full Datasets/vae_erpcore_N2P3_full.csv')

    N170_eeg_c0, N170_eeg_c1, N170_gan_c0, N170_gan_c1, N170_vae_c0, N170_vae_c1 = load_data(f'data/ERPCORE/N170/Full Datasets/erpcore_N170_full_cleaned.csv', 
                                                                                             f'generated_samples/ERPCORE/N170/Full Datasets/gan_erpcore_N170_full_cleaned.csv',
                                                                                             f'generated_samples/ERPCORE/N170/Full Datasets/vae_erpcore_N170_full_cleaned.csv')

    N2PC_eeg_c0, N2PC_eeg_c1, N2PC_gan_c0, N2PC_gan_c1, N2PC_vae_c0, N2PC_vae_c1 = load_data(f'data/ERPCORE/N2PC/Full Datasets/erpcore_N2PC_full_cleaned.csv', 
                                                                                             f'generated_samples/ERPCORE/N2PC/Full Datasets/gan_erpcore_N2PC_full_cleaned.csv',
                                                                                             f'generated_samples/ERPCORE/N2PC/Full Datasets/vae_erpcore_N2PC_full_cleaned.csv')

    #######################################
    ## PLOT ALL
    #######################################

    num_rows = 4

    fig = plt.figure(figsize=(12,num_rows*3))

    #######################################
    ## ERPS
    #######################################

    def plot_ERP(c0, c1, num_item, time, title, num_rows=4):
        ax1 = plt.subplot(num_rows,3, num_item)
        plt.plot(time, c0, axis=0)
        plt.plot(time, c1, axis=0)
        plt.ylabel(r'Voltage ($\mu$V)')
        plt.xlabel('Time (ms)')
        plt.title(title, loc = 'left', x = .02, y=.9)
        #plt.ylim((-2,14))
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)

    plot_ERP(REWP_eeg_c0, REWP_eeg_c1, 1, np.linspace(0,100,100)/100, 'Empirical')
    plot_ERP(REWP_gan_c0, REWP_gan_c1, 2, np.linspace(0,100,100)/100, 'GAN-Synthetic')
    plot_ERP(REWP_vae_c0, REWP_vae_c1, 3, np.linspace(0,100,100)/100, 'VAE-Synthetic')

    plot_ERP(N2P3_eeg_c0, N2P3_eeg_c1, 4, np.linspace(0,100,125)/100, '')
    plot_ERP(N2P3_gan_c0, N2P3_gan_c1, 5, np.linspace(0,100,125)/100, '')
    plot_ERP(N2P3_vae_c0, N2P3_vae_c1, 6, np.linspace(0,100,125)/100, '')

    plot_ERP(N170_eeg_c0, N170_eeg_c1, 7, np.linspace(0,100,128)/100, '')
    plot_ERP(N170_gan_c0, N170_gan_c1, 8, np.linspace(0,100,128)/100, '')
    plot_ERP(N170_vae_c0, N170_vae_c1, 9, np.linspace(0,100,128)/100, '')

    plot_ERP(N2PC_eeg_c0, N2PC_eeg_c1, 10, np.linspace(0,100,128)/100, '')
    plot_ERP(N2PC_gan_c0, N2PC_gan_c1, 11, np.linspace(0,100,128)/100, '')
    plot_ERP(N2PC_vae_c0, N2PC_vae_c1, 12, np.linspace(0,100,128)/100, '')

    #######################################
    ## Save
    #######################################
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, num_rows*4)

    fig.savefig(f'figures/Figure N - ERPCORE_{component}_Evaluations.png', dpi=600)

if __name__ == '__main__':

    #User inputs
    run_TFT = False
    process_synthetic=True

    run_gan = True
    run_vae = False

    component = 'N2PC'
    
    main(run_TFT=run_TFT, process_synthetic=process_synthetic, run_gan=run_gan, run_vae=run_vae, component=component)
