import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import numpy as np

import ipywidgets as wd
from IPython.display import clear_output

###############################################
## CLASSES                                   ##
###############################################

class InteractivePlot:

    def __init__(self, component):

        self.selected_targets = []
        self.available_targets, _ = retrieveData(0, 1)
        self.component = component

        self.select_target_dropdown = wd.Dropdown(options=[' ', 'All']+self.available_targets, value = ' ', description = 'Select', disabled = False)
        self.remove_target_dropdown = wd.Dropdown(options=[' ']+self.selected_targets, value = ' ', description = 'Remove', disabled = False)
        self.centered_toggle = wd.ToggleButton(value=True, description='Centered')

        self.select_target_dropdown.observe(self.change_add)
        self.remove_target_dropdown.observe(self.change_remove)
        self.centered_toggle.observe(self.change_and_refresh)
        self.refresh()

    def refresh(self):
        clear_output()
        prefix_options_select = [' ', 'All'] if self.available_targets else [' ']
        self.select_target_dropdown.options = prefix_options_select+self.available_targets

        prefix_options_remove = [' ', 'All'] if self.selected_targets else [' ']
        self.remove_target_dropdown.options = prefix_options_remove+self.selected_targets

        display(self.select_target_dropdown) #Display the widget for use
        display(self.remove_target_dropdown) #Display the widget for use
        display(self.centered_toggle) #Display the widget for use
        print(f"Selected methods: {', '.join(self.selected_targets)}")
        plot_main(self.component, self.selected_targets, self.centered_toggle.value)

    def change_and_refresh(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.refresh()

    def change_add(self, change):
        if change['type'] == 'change' and change['name'] == 'value' and self.select_target_dropdown.value != ' ':
            if self.select_target_dropdown.value == 'All':
                self.selected_targets, _ = retrieveData(0, 1)
                self.available_targets = []
            else:
                self.selected_targets.append(self.select_target_dropdown.value)
                self.available_targets.remove(self.select_target_dropdown.value)
            self.refresh()

    def change_remove(self, change):
        if change['type'] == 'change' and change['name'] == 'value' and self.remove_target_dropdown.value != ' ':
            if self.remove_target_dropdown.value == 'All':
                self.selected_targets = []
                self.available_targets, _ = retrieveData(0, 1)
            else:
                self.selected_targets.remove(self.remove_target_dropdown.value)
                self.available_targets.append(self.remove_target_dropdown.value)
            self.refresh()

###############################################
## FUNCTIONS                                 ##
###############################################

#Define function to determine filenames
def retrieveData(data, component='N400'): 
    analysis_names = ['Empirical', 'GAN-Augmented', 'Oversampled-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']
    #analysis_names = ['Empirical', 'GAN-Augmented', 'VAE-Augmented', 'Oversampled-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']

    filenames= [
        f'Classification Results/empPredictions_e1_NN.csv',
        f'Classification Results/ganPredictions_e1_NN.csv',
        #f'Classification Results/vaePredictions_e1_NN.csv',
        f'Classification Results/overPredictions_e1_NN.csv',
        f'Classification Results/gausPredictions_e1_NN.csv',
        f'Classification Results/negPredictions_e1_NN.csv',
        f'Classification Results/revPredictions_e1_NN.csv',
        f'Classification Results/smoothPredictions_e1_NN.csv',
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
def loadAndPlot(filename, plotColor, legendName, selected=False, alpha=1, selected_alpha=1, offset=0, center_toggle=True):
    
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
    x_axis_scale = [5, 10, 15, 20]
    if selected:
        selected_offset = offset * center_toggle
        plt.plot(x_axis_scale+selected_offset, meanData, color = plotColor, linewidth = 1, alpha=selected_alpha, label='_nolegend_')
        plt.scatter(x_axis_scale+selected_offset, meanData, label='_nolegend_', color = plotColor, s = 10, alpha=selected_alpha, linewidths=0)
    
    plt.bar(x_axis_scale+offset, meanData, color = plotColor, linewidth = 1, alpha=alpha, width=.5, label = '_nolegend_')
    markers, caps, bars = plt.errorbar(x_axis_scale+offset, meanData, semData, label='_nolegend_', color = plotColor, fmt=' ', linewidth = 1, alpha=alpha)
    [bar.set_alpha(alpha) for bar in bars]
    [cap.set_alpha(alpha) for cap in caps]
        
    return legendName

#Define function to plot difference bars
def plotDiffData(target, reference, analysis_names, filenames):

    target_index = [i for i,f in enumerate(analysis_names) if target == f][0]
    target_filename = filenames[target_index]

    #Load empirical data
    nnDataDS = []
    with open(target_filename) as f:
        [nnDataDS.append(line.split(',')[0:4]) for line in f.readlines()]
    nnDataDS = np.asarray(nnDataDS).astype(int)

    #Determine means of empirical data
    meanDataDS = []
    for ss in np.unique(nnDataDS[:,0]):
        ssIndex = nnDataDS[:,0] == ss
        meanDataDS.append(np.mean(nnDataDS[ssIndex,3]))

    if reference:
        reference_index = [i for i,f in enumerate(analysis_names) if reference == f][0]
        reference_filename = filenames[reference_index]
        #Load synthetic data
        nnDataDSSyn_SynP100 = []
        with open(reference_filename) as f:
            [nnDataDSSyn_SynP100.append(line.split(',')[0:4]) for line in f.readlines()]
        nnDataDSSyn_SynP100 = np.asarray(nnDataDSSyn_SynP100).astype(int)

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
    if reference:
        plt.bar(np.arange(1,5),meanDiff,color='grey',width=.5)
    else:
        plt.bar(np.arange(1,5),meanDataDS,color='grey',width=.5)
    
    '''
    for i in range(len(meanDataDSSyn_SynP100)):
        if meanDiff[i] > 0:
            plt.annotate(str(round(meanDiff[i]))+'%', ([5,10,15,20],meanDiff[i]+.5), ha='center', color='grey', size = 3)
        else:
            plt.annotate(str(round(meanDiff[i]))+'%', ([5,10,15,20],.5), ha='center', color='grey', size = 3) 
    '''  
        
#Main plotting function
def plot_main(component='N400', targets=None, center_toggle=True):
    
    ###############################################
    ## SETUP                                     ##
    ###############################################

    #Determine the sample sizes of interest
    xLabels = [5,10,15,20]
    data = np.arange(1,6)
        
    ###############################################
    ## SETUP FIGURE                              ##
    ###############################################

    #Figure Parameters
    ylims = 75
    xlims = 22.5
    alpha = .6
    alpha_selected = .8
    alpha_targets = .1
    alpha_nontargets = .1
    fontsize = 10

    #Setup
    fig = plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 5})  
    fig.text(0.09, 0.5, 'Prediction Accuracy (%)', va='center', rotation='vertical', fontsize=fontsize)

    ###############################################
    ## PLOT NEURAL NETWORK                       ##
    ###############################################

    #Iterate through all subplots
    for dat in data:
    
        #Signify subplot
        dat_index = dat-1
        ax1 = plt.subplot2grid((5, 4), (dat_index, 0), colspan=3)

        #Create horizontal lines
        axisLevels = np.arange(50,ylims,5)
        for y in axisLevels:
            plt.axhline(y=y, color='k', linestyle=':', alpha=.05, label='_nolegend_')

        #Load and plot data while extracting legend names
        legendNames = []
        analysis_names, filenames = retrieveData(dat, component)
        colors = [f'C{i}' for i in range(10)]
        offsets = (np.arange(len(analysis_names)) - ((len(analysis_names)/2)-.5))/2
        for i, filename in enumerate(filenames):
            if targets:
                if analysis_names[i] in targets:
                    legendNames.append(loadAndPlot(filename, 
                                                colors[i], 
                                                analysis_names[i], 
                                                alpha=alpha_targets, 
                                                selected_alpha=alpha_selected, 
                                                offset=offsets[i], 
                                                center_toggle=center_toggle, 
                                                selected=True))
                else:
                    legendNames.append(loadAndPlot(filename, 
                                                colors[i], 
                                                analysis_names[i], 
                                                alpha=alpha_nontargets, 
                                                offset=offsets[i], 
                                                selected=False))
            else:
                legendNames.append(loadAndPlot(filename, 
                                            colors[i], 
                                            analysis_names[i], 
                                            alpha=alpha, 
                                            offset=offsets[i], 
                                            selected=False))

        #Format plot
        x_axis_scale = [5, 10, 15, 20]
        plt.ylim(45,ylims)
        plt.xlim(2.5, xlims)
        plt.xticks(x_axis_scale, xLabels, fontsize=fontsize-2)
        plt.yticks(np.arange(50,ylims,5), fontsize=fontsize-2)
        #plt.xticks(rotation=90)
        ax1.spines[['right', 'top']].set_visible(False)
        for y in np.arange(5,15,5):
            plt.axhline(y=y, color='k', linestyle=':', alpha=.05, label='_nolegend_')
            
        #Plot legend on last subplot
        if dat == 1:
            legend_elements = []
            for i, legend_name in enumerate(legendNames):
                if targets:
                    if analysis_names[i] in targets:
                        legend_elements.append(Patch(facecolor=colors[i], edgecolor=None, label=legend_name, alpha=alpha_selected))
                    else:
                        legend_elements.append(Patch(facecolor=colors[i], edgecolor=None, label=legend_name, alpha=alpha_nontargets))
                else:
                    legend_elements.append(Patch(facecolor=colors[i], edgecolor=None, label=legend_name, alpha=alpha_selected))

            plt.legend(legend_elements, legendNames, bbox_to_anchor=(.78,1.3), frameon=False, ncol=np.ceil(len(legendNames)/2), fontsize=fontsize-2)
            
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
        
            
        #Add classifier titles
        if (dat == 1):
            ax1.annotate('Neural Network', (2.75,ylims), fontsize = fontsize)
        elif (dat == 2):
            ax1.annotate('Support Vector Machine', (2.75,ylims), fontsize = fontsize)
        elif (dat == 3):
            ax1.annotate('Logistic Regression', (2.75,ylims), fontsize = fontsize)
        elif (dat == 4):
            ax1.annotate('Random Forest', (2.75,ylims), fontsize = fontsize)
        elif (dat == 5):
            ax1.annotate('K-Nearest Neighbors', (2.75,ylims), fontsize = fontsize)
        
        #Add difference subplot if two+ targets are selected
        ax1 = plt.subplot2grid((5, 4), (dat_index, 3), colspan=1)
        if len(targets) == 1:
            plotDiffData(targets[0], None, analysis_names, filenames)
        elif len(targets) >= 2:
            plotDiffData(targets[0], targets[1], analysis_names, filenames)
        ax1.spines[['right', 'top']].set_visible(False)
        plt.xlim(0,5)
        plt.xticks(np.arange(1,5), ['5','10','15','20'], fontsize=fontsize-2)

        if len(targets) >= 2:
            plt.ylim(0,15)
            plt.yticks(np.arange(0,15,5), fontsize=fontsize-2)
        else:
            plt.ylim(50,ylims)
            plt.yticks(np.arange(50,ylims,5), fontsize=fontsize-2)

        if dat == 1:
            if len(targets) == 1:
                ax1.annotate(f'{targets[0]}',(0.55,ylims), fontsize = fontsize)
            elif len(targets) >= 2:
                ax1.annotate(f"{targets[0].replace('-Augmented','')} - {targets[1].replace('-Augmented','')}",(0.55,15), fontsize = fontsize)
            else:
                ax1.annotate(f'Select an Analysis',(0.55,ylims), fontsize = fontsize)
        elif dat == 5:
            plt.xlabel('Sample Size', fontsize=fontsize)


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
