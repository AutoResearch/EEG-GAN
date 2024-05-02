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
def main(run_TFT=True, process_synthetic=True, run_gan=True, run_vae=True):

    print('Loading data...')
    
    ## EMPIRICAL ##

    #Load EEG Data (Participant, Condition, Trial, Electrode, Time1, ...)
    EEG_data = np.genfromtxt(f'data/ERPCORE/N170/Full Datasets/erpcore_N170_full_cleaned.csv', delimiter=',', skip_header=1)[:,1:] #Removes Participant Column
    EEG_data = np.delete(EEG_data, 1, 1) #Delete Unused Column (Trial)

    #Split into conditions
    related_EEG_data = EEG_data[np.r_[EEG_data[:,0]==1],2:]  
    unrelated_EEG_data = EEG_data[np.r_[EEG_data[:,0]==0],2:] 

    if run_gan:
        ## GAN ##
        
        #Load and Process Synthetic Data (Condition, Electrode, Time0, ...)
        gan_fn_0 = f'generated_samples/ERPCORE/N170/Full Datasets/gan_erpcore_N170_full_cleaned_c0.csv'
        gan_fn_1 = f'generated_samples/ERPCORE/N170/Full Datasets/gan_erpcore_N170_full_cleaned_c1.csv'
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
        fftwinganData = fftTempganData[np.r_[fftTempganData[:,0]==0],1:]
        fftlossganData = fftTempganData[np.r_[fftTempganData[:,0]==1],1:]

        #Average data
        avgLossganData = np.mean(lossganData, axis=0)
        avgWinganData = np.mean(winganData, axis=0)

        #Scale synthetic data 
        EEGDataScale = np.max(np.mean(related_EEG_data, axis=0))-np.min(np.mean(related_EEG_data, axis=0)) 
        EEGDataOffset = np.min(np.mean(related_EEG_data, axis=0))
        ganDataScale = np.max(avgLossganData)-np.min(avgLossganData)
        ganDataOffset = np.min(avgLossganData)

        avgLossganData = (((avgLossganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
        avgWinganData = (((avgWinganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

        scaledLossganData = (((lossganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
        scaledWinganData = (((winganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

    if run_vae:
        ## VAE ##
        
        vae_fn_0 = f'generated_samples/ERPCORE/N170/Full Datasets/vae_erpcore_N170_full_c0.csv'
        vae_fn_1 = f'generated_samples/ERPCORE/N170/Full Datasets/vae_erpcore_N170_full_c1.csv'
        vaeData0 = np.genfromtxt(vae_fn_0, delimiter=',', skip_header=1)
        vaeData1 = np.genfromtxt(vae_fn_1, delimiter=',', skip_header=1)
        vaeData = np.vstack((vaeData0, vaeData1))
        vaeData = np.delete(vaeData, 1,1) #Remove the electrode column but we need to keep it

        #Process synthetic data
        fftTempvaeData = vaeData
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
        fftwinvaeData = fftTempvaeData[np.r_[fftTempvaeData[:,0]==0],1:]
        fftlossvaeData = fftTempvaeData[np.r_[fftTempvaeData[:,0]==1],1:]

        #Average data
        avgLossvaeData = np.mean(lossvaeData, axis=0)
        avgWinvaeData = np.mean(winvaeData, axis=0)

        #Scale synthetic data 
        EEGDataScale = np.max(np.mean(related_EEG_data, axis=0))-np.min(np.mean(related_EEG_data, axis=0)) 
        EEGDataOffset = np.min(np.mean(related_EEG_data, axis=0))
        vaeDataScale = np.max(avgLossvaeData)-np.min(avgLossvaeData)
        vaeDataOffset = np.min(avgLossvaeData)

        avgLossvaeData = (((avgLossvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
        avgWinvaeData = (((avgWinvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

        scaledLossvaeData = (((lossvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
        scaledWinvaeData = (((winvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

    print('Data loading complete!')

    ###############################################
    ## TRANSFORM DATA                            ##
    ###############################################

    unrelated_EEG_FFT = frequency_transform(unrelated_EEG_data)
    related_EEG_FFT = frequency_transform(related_EEG_data)
    if run_gan:
        winGANFFT = frequency_transform(fftwinganData)
        lossGANFFT = frequency_transform(fftlossganData)
    if run_vae:
        winVAEFFT = frequency_transform(fftwinvaeData)
        lossVAEFFT = frequency_transform(fftlossvaeData)

    ## Time-Frequency
    speriod = 1/128

    #Time-Frequency Transform
    if run_TFT:
        frex, unrelaed_EEG_TFT = time_frequency_transform(unrelated_EEG_data, speriod=speriod)
        _, related_EEG_TFT = time_frequency_transform(related_EEG_data, speriod=speriod)

        if run_gan:
            _, winGANTFT = time_frequency_transform(winganData, speriod=speriod)
            _, lossGANTFT = time_frequency_transform(lossganData, speriod=speriod)

        if run_vae:
            _, winVAETFT = time_frequency_transform(winvaeData, speriod=speriod)
            _, lossVAETFT = time_frequency_transform(lossvaeData, speriod=speriod)

    if run_TFT:
        EEGTFT = np.mean(unrelaed_EEG_TFT,0) - np.mean(related_EEG_TFT,0)
        EEGTFT = norm(EEGTFT)
        tr_EEGTFT = constrain(EEGTFT, num_bins=5)

        if run_gan:
            GANTFT = np.mean(winGANTFT,0) - np.mean(lossGANTFT,0)
            GANTFT = norm(GANTFT)
            tr_GANTFT = constrain(GANTFT, num_bins=5)

        if run_vae:
            VAETFT = np.mean(winVAETFT,0) - np.mean(lossVAETFT,0)
            VAETFT = norm(VAETFT)
            tr_VAETFT = constrain(VAETFT, num_bins=5)

    #######################################
    ## PLOT ALL
    #######################################

    num_rows = 3

    time  = np.linspace(0,100,128)/100
    fig = plt.figure(figsize=(12,num_rows*3))

    #######################################
    ## ERPS
    #######################################

    ax1 = plt.subplot(num_rows,3,1)
    plt.plot(time, np.mean(unrelated_EEG_data, axis=0))
    plt.plot(time, np.mean(related_EEG_data, axis=0))
    plt.ylabel(r'Voltage ($\mu$V)')
    plt.xlabel('Time (ms)')
    plt.title('Empirical', loc = 'left', x = .02, y=.9)
    #plt.ylim((-2,14))
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)

    if run_gan:
        #Plot synthetic ERPs
        ax2 = plt.subplot(num_rows,3,2)
        plt.plot(time, avgWinganData)
        plt.plot(time, avgLossganData)
        plt.xlabel('Time (ms)')
        plt.title('GAN-Synthetic', loc = 'left', x = .02, y=.9)
        #plt.ylim((-2,14))
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)

    if run_vae:
        #Plot synthetic ERPs
        ax2 = plt.subplot(num_rows,3,3)
        plt.plot(time, avgWinvaeData)
        plt.plot(time, avgLossvaeData)
        plt.xlabel('Time (ms)')
        plt.title('VAE-Synthetic', loc = 'left', x = .02, y=.9)
        #plt.ylim((-2,14))
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.legend(['Win', 'Lose'],frameon=False)
    
    #######################################
    ## FFTs
    #######################################

    ax=fig.add_subplot(num_rows,3,4)
    #plt.plot(norm(np.mean(unrelated_EEG_FFT,0)[:20]-np.mean(related_EEG_FFT,0)[:20], neg=True))
    plt.plot(norm(np.mean(unrelated_EEG_FFT,0)[:20]))
    plt.plot(norm(np.mean(related_EEG_FFT,0)[:20]))

    for i in np.arange(0,20,2):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xticks(np.arange(0,21,2))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Power')
    plt.axhline(0, color='grey',linestyle='--', linewidth=1, alpha=.3)
    

    if run_gan:
        ax=fig.add_subplot(num_rows,3,5)
        #plt.plot(norm(np.mean(winGANFFT,0)[:20]-np.mean(lossGANFFT,0)[:20], neg=True))
        plt.plot(norm(np.mean(winGANFFT,0)[:20]))
        plt.plot(norm(np.mean(lossGANFFT,0)[:20]))
        for i in np.arange(0,20,2):
            plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.xticks(np.arange(0,21,2))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Power')
        plt.axhline(0, color='grey',linestyle='--', linewidth=1, alpha=.3)

    if run_vae:
        ax=fig.add_subplot(num_rows,3,6)
        plt.plot(norm(np.mean(winVAEFFT,0)[:20]-np.mean(lossVAEFFT,0)[:20]))
        for i in np.arange(0,20,2):
            plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.xticks(np.arange(0,21,2))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Normalized Power')
        plt.axhline(0, color='grey',linestyle='--', linewidth=1, alpha=.3)


    #######################################
    ## TFTs
    #######################################
    
    if run_TFT:
        fig.add_subplot(num_rows,3,7)
        im = plt.pcolormesh(time, frex, tr_EEGTFT, cmap='coolwarm')
        plt.ylim([0,15])
        plt.xticks(np.linspace(0,1,7), [int(x) for x in np.linspace(-200,1000,7)])
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (ms)')

        if run_gan:
            fig.add_subplot(num_rows,3,8)
            plt.pcolormesh(time, frex, tr_GANTFT, cmap='coolwarm')
            plt.ylim([0,15])
            plt.xticks(np.linspace(0,1,7), [int(x) for x in np.linspace(-200,1000,7)])
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (ms)')

        if run_vae:
            ax = fig.add_subplot(num_rows,3,9)
            mesh = plt.pcolormesh(time, frex, tr_VAETFT, cmap='coolwarm')
            plt.ylim([0,15])
            plt.xticks(np.linspace(0,1,7), [int(x) for x in np.linspace(-200,1000,7)])
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (ms)')
            cax = ax.inset_axes([0.5, .9, 0.4, 0.04])
            fig.colorbar(mesh, cax=cax, orientation='horizontal')

    #######################################
    ## Save
    #######################################
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, num_rows*4)

    fig.savefig(f'figures/Figure N - ERPCORE_N170_Evaluations Cleaned.png', dpi=600)

if __name__ == '__main__':

    #User inputs
    run_TFT = True
    process_synthetic=True

    run_gan = True
    run_vae = False
    
    main(run_TFT=run_TFT, process_synthetic=process_synthetic, run_gan=run_gan, run_vae=run_vae)
