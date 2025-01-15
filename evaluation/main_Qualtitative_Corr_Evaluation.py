
import numpy as np
import pandas as pd
from scipy import signal

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

def smooth(data, n=10):
    return np.convolve(data, np.ones(n)/n, mode='same')

def load_data(data, gan_data, vae_data, run_gan=True, run_vae=True, process_synthetic=True, select_electrode=None, antisaccade=False):
    
    print('Loading data...')
    print(data)

    #Load EEG Data (Participant, Condition, Trial, Electrode, Time1, ...)
    EEG_data = np.genfromtxt(data, delimiter=',', skip_header=1)
    if antisaccade:
        EEG_data[:,2], EEG_data[:,3] = EEG_data[:,3].copy(), EEG_data[:,2].copy()
    EEG_PIDs = EEG_data[:,0] #Participant IDs
    EEG_data = np.delete(EEG_data, 0, 1) #Delete Participant Column
    EEG_data = np.delete(EEG_data, 1, 1) #Delete Unused Column (Trial)

    #Select Electrode
    if select_electrode:
        EEG_PIDs = EEG_PIDs[np.r_[EEG_data[:,1]==select_electrode]]
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

    return c0_PIDs, c1_PIDs, c0_EEG_data, c1_EEG_data, scaledc0ganData, scaledc1ganData, scaledc0vaeData, scaledc1vaeData

def load_and_corr(filenames, electrode, antisaccade):
    #Define filenames
    electrode = None
    antisaccade = False
    emp_fn, gan_fn, vae_fn = filenames
    _, _, data_eeg_c0, data_eeg_c1, data_gan_c0, data_gan_c1, data_vae_c0, data_vae_c1 = load_data(emp_fn, gan_fn, vae_fn, select_electrode=electrode, antisaccade=antisaccade)
    
    data_eeg_c0 = np.mean(data_eeg_c0, 0)
    data_eeg_c1 = np.mean(data_eeg_c1, 0)
    data_gan_c0 = np.mean(data_gan_c0, 0)
    data_gan_c1 = np.mean(data_gan_c1, 0)
    data_vae_c0 = np.mean(data_vae_c0, 0)
    data_vae_c1 = np.mean(data_vae_c1, 0)

    data_eeg_diff = data_eeg_c1 - data_eeg_c0
    data_gan_diff = data_gan_c1 - data_gan_c0
    data_vae_diff = data_vae_c1 - data_vae_c0

    corr_gan_diff = np.corrcoef(smooth(data_eeg_diff, int(len(data_eeg_diff)*.05)), smooth(data_gan_diff, int(len(data_gan_diff)*.05)))[0,1]
    corr_vae_diff = np.corrcoef(smooth(data_eeg_diff, int(len(data_eeg_diff)*.05)), smooth(data_vae_diff, int(len(data_vae_diff)*.05)))[0,1]

    return {'GAN': np.round(corr_gan_diff,2), 'VAE': np.round(corr_vae_diff,2)}

def main():
        
    filenames = ['data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e1_len100.csv',
                 'generated_samples/Reinforcement Learning/Full Datasets/aegan_ep2000_p500_e1_full.csv', 
                 'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e1_full.csv']
    RLe1_correlations = load_and_corr(filenames, None, False)

    filenames = ['data/Reinforcement Learning/Full Datasets/ganTrialElectrodeERP_p500_e8_len100.csv',
                 'generated_samples/Reinforcement Learning/Full Datasets/aegan_ep2000_p500_e8_full.csv', 
                 'generated_samples/Reinforcement Learning/Full Datasets/vae_p500_e8_full.csv']
    RLe8_correlations = load_and_corr(filenames, 2, False)

    filenames = ['data/Antisaccade/Full Datasets/antisaccade_left_full_cleaned.csv', 
                 'generated_samples/Antisaccade/Full Datasets/gan_antisaccade_full_cleaned.csv',
                 'generated_samples/Antisaccade/Full Datasets/vae_antisaccade_full_cleaned.csv',]
    as_correlations = load_and_corr(filenames, None, True)
    
    filenames = ['data/ERPCORE/N170/Full Datasets/erpcore_N170_full_cleaned.csv', 
                 'generated_samples/ERPCORE/N170/Full Datasets/gan_erpcore_N170_full_cleaned.csv',
                 'generated_samples/ERPCORE/N170/Full Datasets/vae_erpcore_N170_full_cleaned.csv']
    fp_correlations = load_and_corr(filenames, None, False)
    
    filenames = ['data/ERPCORE/N2PC/Full Datasets/erpcore_N2PC_full_cleaned.csv', 
                 'generated_samples/ERPCORE/N2PC/Full Datasets/gan_erpcore_N2PC_full_cleaned.csv',
                 'generated_samples/ERPCORE/N2PC/Full Datasets/vae_erpcore_N2PC_full_cleaned.csv']
    vs_correlations = load_and_corr(filenames, None, False)

    #Turn into dataframe
    correlation_dataframe = pd.DataFrame([RLe1_correlations, RLe8_correlations, as_correlations, fp_correlations, vs_correlations])
    correlation_dataframe.index = ['Reinforcement Learning (E1)', 'Reinforcement Learning (E8)', 'Antisaccade', 'Face Processing', 'Visual Search']
    correlation_dataframe.index = [f'\\textbf{{{index}}}' for index in correlation_dataframe.index]
    correlation_dataframe.columns = [f'\\textbf{{{column}}}' for column in correlation_dataframe.columns]
    correlation_averages = correlation_dataframe.mean()
    correlation_averages = correlation_averages.apply(lambda x: f'\\textbf{{{str(np.round(x, 2))}}}' if isinstance(x, (int, float)) else x)
    correlation_dataframe.loc['blank'] = ['','']
    correlation_dataframe.loc[f'\\textbf{{Average}}'] = correlation_averages
    correlation_dataframe.reset_index(inplace=True)
    correlation_dataframe.loc[correlation_dataframe['index']=='blank', 'index'] = ''
    correlation_dataframe.rename(columns={'index':f'\\textbf{{Dataset}}'}, inplace=True)
    correlation_dataframe = correlation_dataframe.round(2).astype(str)

    caption = """Pearson correlations between the grand averaged difference waveforms of the empirical EEG data and the synthetic data generated by the GAN and VAE models. The difference waveforms were smoothed using a moving average filter with a window size of 5\% of the total number of samples."""
    correlation_dataframe.to_latex('evaluation/grand_averaged_correlations.tex', index=False, caption=caption, label='tab-0', column_format='lcc')

if __name__ == '__main__':
    main()
