###############################################
## LOAD MODULES                              ##
###############################################
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
import pywt
import multiprocessing as mp

##############################################
## CLASSES                                  ##
##############################################
'''
class TFT(mp.Process): 
    def __init__(self, queue, sample, speriod, index = 0, wavelet = 'cmor1.5-1.0'):
        print(f'Job {index} started')
        mp.Process.__init__(self)
        self.queue = queue
        self.sample = sample
        self.speriod = speriod
        self.wavelet = wavelet

    def time_frequency_wavelets(self, sample, speriod, wavelet = 'cmor1.5-1.0'):
        widths = np.geomspace(1, 1024, num=100)
        
        tf, freqs = pywt.cwt(sample, widths, wavelet, sampling_period=speriod)
        tf = np.abs(tf[:-1, :-1]) #Take the absolute value of the wavelet transform

        return freqs, tf

    def run(self):
        #_, tfr = self.time_frequency_wavelets(self.sample, self.speriod, self.wavelet)
        print('done!')
        #self.queue.put(tfr)
        self.queue.put(0)
'''

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

def norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

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
    for i in range(data.shape[0]):
        transformedData[i] = np.abs(scipy.fft.fft(data[i]))[:data.shape[1]//2+1]
    
    return transformedData

def time_frequency_wavelets(sample, srate, wavelet = 'cmor1.5-1.0'):
    widths = np.geomspace(1, 1024, num=100)
    
    tf, freqs = pywt.cwt(sample, widths, wavelet, sampling_period=srate)
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

'''
def time_frequency_transform(data):
        
    speriod = 1/83.33
    queue = mp.Manager().Queue()
    tfrs = [] #Create empty array to store output

    #Create jobs
    [TFT(queue, sample, speriod, index = i).start() for i, sample in enumerate(data)]

    #Get results
    print('4')
    data_size = len(data)
    while data_size > 0:
        print('1')
        tfr = queue.get()
        print('TFR Complete')
        #tfrs.append()
        data_size -= 1

    return tfrs
'''

###############################################
## LOAD AND PROCESS DATA                     ##
###############################################
def main(electrodes, target_electrode):

    print('Loading data...')
    
    ## EMPIRICAL ##

    #Load EEG Data (Participant, Condition, Trial, Electrode, Time1, ...)
    EEGData = np.genfromtxt(f'data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e{electrodes}_len100.csv', delimiter=',', skip_header=1)[:,1:] #Removes Participant Column
    EEGData = np.delete(EEGData, 1, 1) #Delete Unused Column (Trial)

    #Split into conditions
    lossEEGData = EEGData[np.r_[EEGData[:,0]==1],1:]  
    winEEGData = EEGData[np.r_[EEGData[:,0]==0],1:] 
    lossEEGData = lossEEGData[lossEEGData[:,0]==target_electrode,1:]
    winEEGData = winEEGData[winEEGData[:,0]==target_electrode,1:]

    ## GAN ##

    #Load and Process Synthetic Data (Condition, Electrode, Time0, ...)
    gan_fn_0 = f'generated_samples/Reinforcement Learning/Full Datasets/aegan_ep2000_p500_e{electrodes}_full_c0.csv'
    gan_fn_1 = f'generated_samples/Reinforcement Learning/Full Datasets/aegan_ep2000_p500_e{electrodes}_full_c1.csv'
    ganData0 = np.genfromtxt(gan_fn_0, delimiter=',', skip_header=1)
    ganData1 = np.genfromtxt(gan_fn_1, delimiter=',', skip_header=1)
    ganData = np.vstack((ganData0, ganData1))
    ganData = ganData[np.r_[ganData[:,1]==target_electrode],:] 
    ganData = np.delete(ganData, 1,1) #Remove the electrode column but we need to keep it

    #Process synthetic data
    tempganData = filterEEG(ganData[:,1:])
    tempganData = baselineCorrect(tempganData)

    #Create and populate new array for processed synthetic data
    processedganData = np.zeros((len(tempganData),101))
    processedganData[:,0] = ganData[:,0]
    processedganData[:,1:] = np.array(tempganData)

    #Split into conditions
    lossganData = processedganData[np.r_[processedganData[:,0]==1],1:]
    winganData = processedganData[np.r_[processedganData[:,0]==0],1:]

    #Average data
    avgLossganData = np.mean(lossganData, axis=0)
    avgWinganData = np.mean(winganData, axis=0)

    #Scale synthetic data 
    EEGDataScale = np.max(np.mean(lossEEGData, axis=0))-np.min(np.mean(lossEEGData, axis=0)) 
    EEGDataOffset = np.min(np.mean(lossEEGData, axis=0))
    ganDataScale = np.max(avgLossganData)-np.min(avgLossganData)
    ganDataOffset = np.min(avgLossganData)

    avgLossganData = (((avgLossganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset
    avgWinganData = (((avgWinganData-ganDataOffset)/ganDataScale)*EEGDataScale)+EEGDataOffset

    ## VAE ##

    vae_fn_0 = f'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e{electrodes}_full_c0.csv'
    vae_fn_1 = f'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e{electrodes}_full_c1.csv'
    vaeData0 = np.genfromtxt(vae_fn_0, delimiter=',', skip_header=1)
    vaeData1 = np.genfromtxt(vae_fn_1, delimiter=',', skip_header=1)
    vaeData = np.vstack((vaeData0, vaeData1))
    vaeData = vaeData[np.r_[vaeData[:,1]==target_electrode],:]
    vaeData = np.delete(vaeData, 1,1) #Remove the electrode column but we need to keep it

    #Process synthetic data
    tempvaeData = filterEEG(vaeData[:,1:])
    tempvaeData = baselineCorrect(tempvaeData)

    #Create and populate new array for processed synthetic data
    processedvaeData = np.zeros((len(tempvaeData),101))
    processedvaeData[:,0] = vaeData[:,0]
    processedvaeData[:,1:] = np.array(tempvaeData)

    #Split into conditions
    lossvaeData = processedvaeData[np.r_[processedvaeData[:,0]==1],1:]
    winvaeData = processedvaeData[np.r_[processedvaeData[:,0]==0],1:]

    #Average data
    avgLossvaeData = np.mean(lossvaeData, axis=0)
    avgWinvaeData = np.mean(winvaeData, axis=0)

    #Scale synthetic data 
    EEGDataScale = np.max(np.mean(lossEEGData, axis=0))-np.min(np.mean(lossEEGData, axis=0)) 
    EEGDataOffset = np.min(np.mean(lossEEGData, axis=0))
    vaeDataScale = np.max(avgLossvaeData)-np.min(avgLossvaeData)
    vaeDataOffset = np.min(avgLossvaeData)

    avgLossvaeData = (((avgLossvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset
    avgWinvaeData = (((avgWinvaeData-vaeDataOffset)/vaeDataScale)*EEGDataScale)+EEGDataOffset

    print('Data loading complete!')

    ###############################################
    ## TRANSFORM DATA                            ##
    ###############################################

    winEEGFFT = frequency_transform(winEEGData)
    lossEEGFFT = frequency_transform(lossEEGData)
    winGANFFT = frequency_transform(winganData)
    lossGANFFT = frequency_transform(lossganData)
    winVAEFFT = frequency_transform(winvaeData)
    lossVAEFFT = frequency_transform(lossvaeData)

    ## Time-Frequency
    speriod = 1/83.33

    #Time-Frequency Transform
    frex, winEEGTFT = time_frequency_transform(winEEGData, speriod=speriod)
    _, lossEEGTFT = time_frequency_transform(lossEEGData, speriod=speriod)
    _, winGANTFT = time_frequency_transform(winganData, speriod=speriod)
    _, lossGANTFT = time_frequency_transform(lossganData, speriod=speriod)
    _, winVAETFT = time_frequency_transform(winvaeData, speriod=speriod)
    _, lossVAETFT = time_frequency_transform(lossvaeData, speriod=speriod)

    '''
    import multiprocessing as mp
    results = []
    data_list = [winEEGData, lossEEGData, winganData, lossganData, winvaeData, lossvaeData]
    data_names = ['winEEGData', 'lossEEGData', 'winganData', 'lossganData', 'winvaeData', 'lossvaeData']
    for data, data_name in zip(data_list, data_names):
        p = mp.Process(target=time_frequency_transform, args=(data, speriod, data_name))
        p.start()
        results.append(p)
    '''

    EEGTFT = np.mean(winEEGTFT,0) - np.mean(lossEEGTFT,0)
    GANTFT = np.mean(winGANTFT,0) - np.mean(lossGANTFT,0)
    VAETFT = np.mean(winVAETFT,0) - np.mean(lossVAETFT,0)

    #change EEGTFT to range from red to blue in imshow
    EEGTFT = norm(EEGTFT)
    GANTFT = norm(GANTFT)
    VAETFT = norm(VAETFT)

    tr_EEGTFT = constrain(EEGTFT, num_bins=5)
    tr_GANTFT = constrain(GANTFT, num_bins=5)
    tr_VAETFT = constrain(VAETFT, num_bins=5)

    #######################################
    ## PLOT ALL
    #######################################

    time  = np.linspace(0,83.3,100)/83.3
    fig = plt.figure(figsize=(12,12))

    #######################################
    ## ERPS
    #######################################

    ax1 = plt.subplot(3,3,1)
    plt.plot(time, np.mean(winEEGData, axis=0))
    plt.plot(time, np.mean(lossEEGData, axis=0))
    plt.ylabel(r'Voltage ($\mu$V)')
    plt.xlabel('Time (ms)')
    plt.title('Empirical', loc = 'left', x = .02, y=.9)
    plt.ylim((-2,14))
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)

    #Plot synthetic ERPs
    ax2 = plt.subplot(3,3,2)
    plt.plot(time, avgWinganData)
    plt.plot(time, avgLossganData)
    plt.xlabel('Time (ms)')
    plt.title('GAN-Synthetic', loc = 'left', x = .02, y=.9)
    plt.ylim((-2,14))
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    #Plot synthetic ERPs
    ax2 = plt.subplot(3,3,3)
    plt.plot(time, avgWinvaeData)
    plt.plot(time, avgLossvaeData)
    plt.xlabel('Time (ms)')
    plt.title('VAE-Synthetic', loc = 'left', x = .02, y=.9)
    plt.ylim((-2,14))
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.legend(['Win', 'Lose'],frameon=False)

    #######################################
    ## FFTs
    #######################################

    ax=fig.add_subplot(3,3,4)
    plt.plot(norm(np.mean(winEEGFFT,0)[:10]-np.mean(lossEEGFFT,0)[:10]))
    for i in np.arange(0,10,2):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xticks(np.arange(0,10,2))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Power')

    ax=fig.add_subplot(3,3,5)
    plt.plot(norm(np.mean(winGANFFT,0)[:10]-np.mean(lossGANFFT,0)[:10]))
    for i in np.arange(0,10,2):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xticks(np.arange(0,10,2))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Power')

    ax=fig.add_subplot(3,3,6)
    plt.plot(norm(np.mean(winVAEFFT,0)[:10]-np.mean(lossVAEFFT,0)[:10]))
    for i in np.arange(0,10,2):
        plt.axvline(x=i, color='grey', linestyle='--', alpha=.3)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xticks(np.arange(0,10,2))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Power')

    #######################################
    ## TFTs
    #######################################

    fig.add_subplot(3,3,7)
    im = plt.pcolormesh(time, frex, tr_EEGTFT, cmap='coolwarm')
    plt.ylim([0,15])
    plt.xticks(np.linspace(0,1,7), [int(x) for x in np.linspace(-200,1000,7)])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')

    fig.add_subplot(3,3,8)
    plt.pcolormesh(time, frex, tr_GANTFT, cmap='coolwarm')
    plt.ylim([0,15])
    plt.xticks(np.linspace(0,1,7), [int(x) for x in np.linspace(-200,1000,7)])
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')

    ax = fig.add_subplot(3,3,9)
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
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.savefig(f'Figure N - Evaluations num_electrodes {electrodes} electrode {target_electrode}.png', dpi=600)

if __name__ == '__main__':

    electrodes = [1]
    
    for electrode in electrodes:
        for target_electrode in range(1, electrode+1):
            main(electrode, target_electrode)
