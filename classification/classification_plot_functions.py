import matplotlib.pyplot as plt
import numpy as np

###############################################
## FUNCTIONS                                 ##
###############################################

#Define function to determine filenames
def retrieveData(data, electrodes): 
    analysis_names = ['Empirical', 'GAN-Augmented', 'VAE-Augmented', 'Oversampled-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']
    #analysis_names = ['Empirical', 'GAN-Augmented', 'VAE-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']

    filenames= [
        f'Classification Results/empiricalPredictions_e{electrodes}_NN.csv',
        f'Classification Results/augmentedPredictions_e{electrodes}_NN.csv',
        f'Classification Results/vaePredictions_e{electrodes}_NN.csv',
        f'Classification Results/overPredictions_e{electrodes}_NN.csv',
        f'Classification Results/gausPredictions_e{electrodes}_NN.csv',
        f'Classification Results/negPredictions_e{electrodes}_NN.csv',
        f'Classification Results/revPredictions_e{electrodes}_NN.csv',
        f'Classification Results/smoothPredictions_e{electrodes}_NN.csv',
    ]

    if data == 1:
        filenames = [filename.replace('_NN', '_SVM') for filename in filenames]
    if data == 2:
        filenames = [filename.replace('_NN', '_LR') for filename in filenames]
    if data == 3:
        filenames = [filename.replace('_NN', '_RF') for filename in filenames]
    if data == 4:
        filenames = [filename.replace('_NN', '_KNN') for filename in filenames]

    ##Full Time-Series
    return analysis_names, filenames


#Define function to load and plot data
def loadAndPlot(filename, plotColor, legendName, selected=False, alpha=1, selected_alpha=1, offset=0):
    
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
    if selected:
        plt.plot(np.arange(1,8), meanData, color = plotColor, linewidth = 1, alpha=selected_alpha, label='_nolegend_')
        plt.scatter(np.arange(1,8), meanData, label='_nolegend_', color = plotColor, s = 10, alpha=selected_alpha, linewidths=0)
    
    plt.bar(np.arange(1,8)+offset, meanData, color = plotColor, linewidth = 1, alpha=alpha, width=.1, label = legendName)
    markers, caps, bars = plt.errorbar(np.arange(1,8)+offset, meanData, semData, label='_nolegend_', color = plotColor, fmt=' ', linewidth = 1, alpha=alpha)
    [bar.set_alpha(alpha) for bar in bars]
    [cap.set_alpha(alpha) for cap in caps]
        
    return legendName

#Define function to plot difference bars
def plotDiffData():
    
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
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],meanDiff[i]+.5), ha='center', color='grey', size = 3)
        else:
            ax2.annotate(str(round(meanDiff[i]))+'%', (xLabels[i],.5), ha='center', color='grey', size = 3)   


#Main plotting function
def plot_main(target=None, referent=None):
    ###############################################
    ## SETUP                                     ##
    ###############################################

    #Determine the sample sizes of interest
    xLabels = [5,10,15,20,30,60,100]
    electrodes = 1
    combined = False #Whether to add multiple augmented data to a single plot
    data = np.arange(1,6)
        
    ###############################################
    ## SETUP FIGURE                              ##
    ###############################################

    #Figure Parameters
    ylims = 80
    alpha = .6
    alpha_selected = .8
    alpha_targets = .1
    alpha_nontargets = .1
    fontsize = 10

    #Setup
    #fig = plt.figure(figsize=(24, 3), dpi=600)
    fig = plt.figure(figsize=(10, 7.5))

    fig.subplots_adjust(hspace=.3)
    plt.rcParams.update({'font.size': 5})  
    fig.text(0.09, 0.5, 'Prediction Accuracy (%)', va='center', rotation='vertical', fontsize=fontsize)

    ###############################################
    ## PLOT NEURAL NETWORK                       ##
    ###############################################

    #Iterate through all subplots
    for dat in data:
        
        #Signify subplot
        ax1 = plt.subplot(5,1,dat)

        #Create horizontal lines
        axisLevels = np.arange(50,ylims,5)
        for y in axisLevels:
            plt.axhline(y=y, color='k', linestyle=':', alpha=.05, label='_nolegend_')

        #Load and plot data while extracting legend names
        legendNames = []
        analysis_names, filenames = retrieveData(dat, electrodes)
        colors = [f'C{i}' for i in range(10)]
        offsets = (np.arange(len(analysis_names)) - ((len(analysis_names)/2)-.5))/10
        for i, filename in enumerate(filenames):
            if target or referent:
                if analysis_names[i] == target or analysis_names[i] == referent:
                    legendNames.append(loadAndPlot(filename, colors[i], analysis_names[i], alpha=alpha_targets, selected_alpha=alpha_selected, offset=offsets[i], selected=True))
                else:
                    legendNames.append(loadAndPlot(filename, colors[i], analysis_names[i], alpha=alpha_nontargets, offset=offsets[i], selected=False))
            else:
                legendNames.append(loadAndPlot(filename, colors[i], analysis_names[i], alpha=alpha, offset=offsets[i], selected=False))

        #Formal plot
        plt.ylim(45,ylims)
        plt.xlim(0.5, 7.5)
        plt.xticks(np.arange(1,8), xLabels, fontsize=fontsize-2)
        plt.yticks(np.arange(50,ylims,5), fontsize=fontsize-2)
        #plt.xticks(rotation=90)
        ax1.spines[['right', 'top']].set_visible(False)
            
        #Plot legend on last subplot
        if dat == 1:
            plt.legend(legendNames, bbox_to_anchor=(.4,1.05), frameon=False, ncol=np.ceil(len(legendNames)/2), fontsize=fontsize-2)
            
        #Plot y label on left subplots
        #plt.ylabel('Prediction Accuracy (%)', fontsize=12)
            
        #Plot x label on bottom subplots
        if dat == 5:    
            plt.xlabel('Sample Size', fontsize=fontsize)
        else:
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        
        #Add data type titles
        if dat == 1:
            ax1.annotate('Full Time Series',(0.55,ylims), fontsize = fontsize)
        elif dat == 6:
            ax1.annotate('Extracted Features',(0.55,ylims), fontsize = fontsize)
        elif dat == 11:
            ax1.annotate('Autoencoder Features',(0.55,ylims), fontsize = fontsize)
            
        #Add classifier titles
        if (dat == 1):
            ax1.annotate('Neural Network', (0.55,ylims-5), fontsize = fontsize-2)
        elif (dat == 2):
            ax1.annotate('Support Vector Machine', (0.55,ylims-5), fontsize = fontsize-2)
        elif (dat == 3):
            ax1.annotate('Logistic Regression', (0.55,ylims-5), fontsize = fontsize-2)
        elif (dat == 4):
            ax1.annotate('Random Forest', (0.55,ylims-5), fontsize = fontsize-2)
        elif (dat == 5):
            ax1.annotate('K-Nearest Neighbors', (0.55,ylims-5), fontsize = fontsize-2)

    ###############################################
    ## SAVE PLOT                                 ##
    ###############################################
    plt.show()

    '''
    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    if combined:
        fig.savefig(f'classification/Figures/Figure N - Combined Classification Results (e{electrodes}).png', dpi=600, facecolor='white', edgecolor='none')
    else:
        fig.savefig(f'classification/Figures/Figure N - COMPARISON Classification Results (e{electrodes}).png', dpi=600, facecolor='white', edgecolor='none')
    '''
