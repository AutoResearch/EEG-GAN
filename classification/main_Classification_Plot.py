###############################################
## IMPORT MODULES                            ##
###############################################
import matplotlib.pyplot as plt
import numpy as np
import os

###############################################
## FUNCTIONS                                 ##
###############################################

#Define function to determine filenames
def retrieveData(prefix, classifier, electrodes=1): 

    augData = f'{prefix}/ganPredictions_e{electrodes}_{classifier}.csv'
    empData = f'{prefix}/empPredictions_e{electrodes}_{classifier}.csv'

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
    if len(np.unique(data[:,0])) == 7:
        xLabels = [5,10,15,20,25,30,35]
    else:
        xLabels = [5,10,15,20]

    plt.plot(xLabels, meanData, color = plotColor, linewidth = 2, alpha=alpha)
    plt.scatter(xLabels,meanData,label='_nolegend_', color = plotColor, s = 10, alpha=alpha, linewidths=0)
    markers, caps, bars = plt.errorbar(xLabels, meanData, semData, label='_nolegend_', color = plotColor, fmt=' ', linewidth = 1, alpha=alpha)
    [bar.set_alpha(alpha) for bar in bars]
    [cap.set_alpha(alpha) for cap in caps]
        
    return legendName

#Define function to plot difference bars
def plotDiffData(empData, augData, ax2, xLabels):
    
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
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],meanDiff[i]+.6), ha='center', color='grey', size = 10)
        else:
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],.6), ha='center', color='grey', size = 10)   
    
def main():
    ###############################################
    ## SETUP FIGURE                              ##
    ###############################################

    num_rows = 5
    fig = plt.figure(figsize=(24, 3*num_rows), dpi=600)
    fig.subplots_adjust(hspace=.3, bottom=0.2)
    plt.rcParams.update({'font.size': 5})  

    ylims = 75
    alpha = 1
    fontsize = 6

    ###############################################
    ## PLOT NEURAL NETWORK                       ##
    ###############################################

    datasets = ['classification/Reinforcement Learning/Classification Results',
                'classification/Reinforcement Learning/Classification Results',
                'classification/Antisaccade/Classification Results',
                'classification/ERPCORE/N170/Classification Results',
                'classification/ERPCORE/N2PC/Classification Results']
    
    classifiers = ['NN','SVM','LR','RF','KNN']

    for dataset_index, dataset in enumerate(datasets):

        if dataset_index < 3:
            xLabels = [5,10,15,20,25,30,35]
        else:
            xLabels = [5,10,15,20]

        #Iterate through all subplots
        for classifier_index, classifier in enumerate(classifiers):
            
            #Signify subplot
            ax1 = plt.subplot(num_rows,5,(classifier_index+1)+(5*dataset_index))

            #Signify item number
            num_item = (classifier_index+1)+(5*dataset_index)

            #Load and plot data while extracting legend names
            legendNames = []
            if False: #dataset_index == 1: #TODO: ADD THIS BACK IN
                empData, augData = retrieveData(dataset, classifier, electrodes=8)
            else:
                empData, augData = retrieveData(dataset, classifier)

            legendNames.append(loadAndPlot(augData,'C0','GAN-Augmented',alpha=alpha))
            legendNames.append(loadAndPlot(empData,'C1','Empirical',alpha=alpha))

            #Create horizontal lines
            axisLevels = np.arange(50,ylims,5)
            for y in axisLevels:
                plt.axhline(y=y, color='k', linestyle=':', alpha=.1)
                
            #Formal plot
            plt.ylim(40,ylims)
            plt.xlim(2.5,22.5)
            plt.xticks(xLabels, fontsize = 10)
            #change xtick labels to ['5','10','15','20','30','60','100']
            if dataset_index < 3:
                plt.xticks(np.arange(5,40,5), ['5','10','15','20','30','60','100'], fontsize = 10)
                plt.xlim(2.5,37.5)

            plt.yticks(np.arange(50,ylims,5), fontsize = 10)
            ax1.spines[['right', 'top']].set_visible(False)
                
            #Plot legend on last subplot
            if num_item == 5:
                plt.legend(legendNames, bbox_to_anchor=(.9,1.11), frameon=False, fontsize=10)
                
            #Plot y label on left subplot
            if (num_item == 1 or num_item == 6 or num_item == 11 or num_item == 16 or num_item == 21):
                plt.ylabel('Prediction Accuracy (%)', fontsize = 12)
                
            #Plot x label on bottom subplots   
            if (num_item > 20):
                plt.xlabel('Sample Size', fontsize = 12)
                
            #Add classifier titles
            if (num_item == 1):
                plt.text(0.5, 1.3, 'Neural\nNetwork', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            elif (num_item == 2):
                plt.text(0.5, 1.3, 'Support\nVector\nMachine', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            elif (num_item == 3):
                plt.text(0.5, 1.3, 'Logistic\nRegression', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            elif (num_item == 4):
                plt.text(0.5, 1.3, 'Random\nForest', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
            elif (num_item == 5):
                plt.text(0.5, 1.3, 'K-Nearest\nNeighbors', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')

            #Row labels
            if num_item == 1:
                plt.text(-.25, 1.1, 'Reinforcement\nLearning (E1)', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
            elif num_item == 6:
                plt.text(-.25, 1.1, 'Reinforcement\nLearning (E8)', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
            elif num_item == 11:
                plt.text(-.25, 1.1, 'Anti-Saccade', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
            elif num_item == 16:
                plt.text(-.25, 1.1, 'Face Perception', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
            elif num_item == 21:
                plt.text(-.25, 1.1, 'Visual Search', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes, fontsize=12, fontweight='bold')
        
            #Add difference bars
            ax2 = ax1.twinx()  
            plotDiffData(empData, augData, ax2, xLabels)
            
            #Format difference bars
            ax2.set_ylim(0,90)
            ax2.spines[['right', 'top']].set_visible(False)
            ax2.set_yticks([])
        
    ###############################################
    ## SAVE PLOT                                 ##
    ###############################################

    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(16, 16)
    plt.subplots_adjust(left=0.05)
    fig.savefig(f'figures/Figure 3 - GAN Classification Results.png', dpi=600, facecolor='white', edgecolor='none')

if __name__ == '__main__':  
    main()