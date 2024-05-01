###############################################
## IMPORT MODULES                            ##
###############################################
import matplotlib.pyplot as plt
import numpy as np

###############################################
## FUNCTIONS                                 ##
###############################################

#Define function to determine filenames
def retrieveData(data): 

    ##Full Time-Series
    if data == 1:
        augData = f'classification/antisaccade/Classification Results/ganPredictions_e1_NN.csv'
        empData = f'classification/antisaccade/Classification Results/empPredictions_e1_NN.csv'
    if data == 2:
        augData = f'classification/antisaccade/Classification Results/ganPredictions_e1_SVM.csv'
        empData = f'classification/antisaccade/Classification Results/empPredictions_e1_SVM.csv'
    if data == 3:
        augData = f'classification/antisaccade/Classification Results/ganPredictions_e1_LR.csv'
        empData = f'classification/antisaccade/Classification Results/empPredictions_e1_LR.csv'
    if data == 4:
        augData = f'classification/antisaccade/Classification Results/ganPredictions_e1_RF.csv'
        empData = f'classification/antisaccade/Classification Results/empPredictions_e1_RF.csv'
    if data == 5:
        augData = f'classification/antisaccade/Classification Results/ganPredictions_e1_KNN.csv'
        empData = f'classification/antisaccade/Classification Results/empPredictions_e1_KNN.csv'

    return empData, augData


#Define function to load and plot data
def loadAndPlot(filename, plotColor, legendName, alpha=1):
    
    #Load data
    data = []
    with open(filename) as f:
        [data.append(line.split(',')[0:4]) for line in f.readlines()]
    data = np.asarray(data).astype(int)
    
    #Process descriptive statistics of means and standard errors of the mean
    meanData = []
    semData = []
    for ss in np.unique(data[:,0]):
        ssIndex = data[:,0] == ss #Sample size
        meanData.append(np.mean(data[ssIndex,3])) #Mean
        semData.append((np.std(data[ssIndex,3]))/np.sqrt(len(data[ssIndex,3]))) #Standard error of the mean
        
    #Plot Data
    plt.plot(np.unique(data[:,0]), meanData, color = plotColor, linewidth = 1, alpha=alpha)
    plt.scatter(np.unique(data[:,0]),meanData,label='_nolegend_', color = plotColor, s = 10, alpha=alpha, linewidths=0)
    markers, caps, bars = plt.errorbar(np.unique(data[:,0]), meanData, semData, label='_nolegend_', color = plotColor, fmt=' ', linewidth = 1, alpha=alpha)
    [bar.set_alpha(alpha) for bar in bars]
    [cap.set_alpha(alpha) for cap in caps]
        
    return legendName

#Define function to plot difference bars
def plotDiffData(empData, augData, ax2):
    
    #Load empirical data
    nnDataDS = []
    with open(empData) as f:
        [nnDataDS.append(line.split(',')[0:4]) for line in f.readlines()]
    nnDataDS = np.asarray(nnDataDS).astype(int)

    #Load synthetic data
    nnDataDSSyn_SynP100 = []
    with open(augData) as f:
        [nnDataDSSyn_SynP100.append(line.split(',')[0:4]) for line in f.readlines()]
    nnDataDSSyn_SynP100 = np.asarray(nnDataDSSyn_SynP100).astype(int)

    #Determine means of empirical data
    meanDataDS = []
    for ss in np.unique(nnDataDS[:,0]):
        ssIndex = nnDataDS[:,0] == ss
        meanDataDS.append(np.mean(nnDataDS[ssIndex,3]))

    #Determine means of synthetic data
    meanDataDSSyn_SynP100 = []
    for ss in np.unique(nnDataDSSyn_SynP100[:,0]):
        ssIndex = nnDataDSSyn_SynP100[:,0] == ss
        meanDataDSSyn_SynP100.append(np.mean(nnDataDSSyn_SynP100[ssIndex,3]))

    #Determine mean differences
    meanDiff = []
    for ss in range(len(meanDataDSSyn_SynP100)):
        meanDiff.append(meanDataDSSyn_SynP100[ss]-meanDataDS[ss])
      
    #Plot mean differences as bars with annotated labels  
    ax2.bar(xLabels,meanDiff,color='grey',width=4)
    for i in range(len(meanDataDSSyn_SynP100)):
        if meanDiff[i] > 0:
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],meanDiff[i]+.6), ha='center', color='grey', size = 6)
        else:
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],.6), ha='center', color='grey', size = 6)   
    
def main(xLabels, data):
    ###############################################
    ## SETUP FIGURE                              ##
    ###############################################

    fig = plt.figure(figsize=(24, 3), dpi=600)
    fig.subplots_adjust(hspace=.3, bottom=0.2)
    plt.rcParams.update({'font.size': 5})  

    ylims = 75
    alpha = 1
    fontsize = 6

    ###############################################
    ## PLOT NEURAL NETWORK                       ##
    ###############################################

    #Iterate through all subplots
    for dat in data:
        
        #Signify subplot
        ax1 = plt.subplot(1,5,dat)

        #Load and plot data while extracting legend names
        legendNames = []
        empData, augData = retrieveData(dat)
        legendNames.append(loadAndPlot(augData,'C0','GAN',alpha=alpha))
        legendNames.append(loadAndPlot(empData,'C1','Empirical',alpha=alpha))

        #Create horizontal lines
        axisLevels = np.arange(50,ylims,5)
        for y in axisLevels:
            plt.axhline(y=y, color='k', linestyle=':', alpha=.1)
            
        #Formal plot
        plt.ylim(40,ylims)
        plt.xlim(2.5,22.5)
        plt.xticks(xLabels)
        plt.yticks(np.arange(50,ylims,5))
        ax1.spines[['right', 'top']].set_visible(False)
            
        #Plot legend on last subplot
        if dat == 5:
            plt.legend(legendNames, bbox_to_anchor=(.9,1.11), frameon=False)
            
        #Plot y label on left subplot
        if (dat == 1):
            plt.ylabel('Prediction Accuracy (%)', fontsize = fontsize)
            
        #Plot x label on bottom subplots   
        plt.xlabel('Sample Size', fontsize = fontsize)
            
        #Add classifier titles
        if (dat == 1):
            ax1.annotate('Neural Network', (3.2,ylims-2.5), fontsize = fontsize)
        elif (dat == 2):
            ax1.annotate('Support Vector Machine', (3.2,ylims-2.5), fontsize = fontsize)
        elif (dat == 3):
            ax1.annotate('Logistic Regression', (3.2,ylims-2.5), fontsize = fontsize)
        elif (dat == 4):
            ax1.annotate('Random Forest', (3.2,ylims-2.5), fontsize = fontsize)
        elif (dat == 5):
            ax1.annotate('K-Nearest Neighbors', (3.2,ylims-2.5), fontsize = fontsize)

        #Add difference bars
        ax2 = ax1.twinx()  
        plotDiffData(empData, augData, ax2)
        
        #Format difference bars
        ax2.set_ylim(0,90)
        ax2.spines[['right', 'top']].set_visible(False)
        ax2.set_yticks([])
        
    ###############################################
    ## SAVE PLOT                                 ##
    ###############################################
    fig = plt.gcf()
    fig.set_size_inches(16, 2)
    fig.savefig(f'figures/Figure N - GAN Classification Results (Antisaccade).png', dpi=600, facecolor='white', edgecolor='none')

if __name__ == '__main__':

    #Define variables
    xLabels = [5,10,15,20]
    data = np.arange(1,6)

    main(xLabels, data)