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

    def __init__(self):

        #Setup initial variables
        self.targets = []
        self.target_labels = []
        self.available_targets, _ = retrieveData(0, 1)
        self.base_target = None
        self.ref_target = None

        #Setup dataset information
        self.datasets = ['Reinforcement Learning (1E)', 
                         'Reinforcement Learning (8E)', 
                         'Antisaccade', 
                         'Face Perception', 
                         'Visual Search']
        self.filename_prefixes = ['Reinforcement Learning/', 
                     'Reinforcement Learning/',
                     'Antisaccade/',
                     'ERPCORE/N170/',
                     'ERPCORE/N2PC/']
        self.x_maxes = [100, 100, 100, 20, 20]
        self.electrodes = [1, 8, 1, 1, 1]

        self.filename_prefix = 'Reinforcement Learning/'
        self.x_max = 100
        self.electrode = 1

        #Setup widgets
        style = {'description_width': '100px'}
        self.select_dataset_dropdown = wd.Dropdown(layout={'width': 'max-content'}, options=self.datasets, value = self.datasets[0], description = ' ', disabled = False, style=style)
        self.targets_checkboxes = [wd.Checkbox(value=False, description=avail_target) for avail_target in self.available_targets]
        self.select_target_row = wd.VBox([wd.HBox(self.targets_checkboxes[:4]), wd.HBox(self.targets_checkboxes[4:])])
        self.centered_toggle = wd.Checkbox(value=False, description='Centered')
        self.selected_base_dropdown = wd.Dropdown(options=[' '] + self.target_labels, value = ' ', description = ' ', disabled = False, style=style)
        self.selected_reference_dropdown = wd.Dropdown(options=[' '] + self.target_labels, value = ' ', description = ' ', disabled = False, style=style)

        for i in range(len(self.targets_checkboxes)):
            self.targets_checkboxes[i].observe(self.change_targets) 
        self.select_dataset_dropdown.observe(self.change_dataset)
        self.centered_toggle.observe(self.change_and_refresh)
        self.selected_base_dropdown.observe(self.select_comp)
        self.selected_reference_dropdown.observe(self.select_comp)

        #Display plot
        self.refresh()

    def refresh(self):
        clear_output()

        #Update target labels
        base_val = self.selected_base_dropdown.value
        ref_val = self.selected_reference_dropdown.value
        self.selected_base_dropdown.options=[' '] + self.target_labels
        self.selected_reference_dropdown.options=[' '] + self.target_labels
        self.selected_base_dropdown.value = base_val
        self.selected_reference_dropdown.value = ref_val

        #Setup controls
        header_style = {'description_width': '150px'}
        header1 = wd.HTML(description="𝐒𝐞𝐥𝐞𝐜𝐭 𝐃𝐚𝐭𝐚𝐬𝐞𝐭", value="", style=header_style)
        header2 = wd.HTML(description="𝐒𝐞𝐥𝐞𝐜𝐭 𝐀𝐧𝐚𝐥𝐲𝐬𝐞𝐬", value="", style=header_style)
        header3 = wd.HTML(description="𝐒𝐞𝐥𝐞𝐜𝐭 𝐂𝐨𝐦𝐩𝐚𝐫𝐢𝐬𝐨𝐧𝐬", value="", style=header_style)
        header4 = wd.HTML(description="𝐒𝐞𝐥𝐞𝐜𝐭 𝐅𝐨𝐫𝐦𝐚𝐭𝐭𝐢𝐧𝐠", value="", style=header_style)
        controls = wd.GridBox(children=[header1, self.select_dataset_dropdown, header2, self.select_target_row, header3, self.selected_base_dropdown, self.selected_reference_dropdown, header4, self.centered_toggle],
        layout=wd.Layout(
            grid_template_rows='auto auto auto auto auto auto auto auto auto',
            grid_template_areas='''
            "header1"
            "drop1"
            "header2"
            "box1"
            "header3"
            "drop2"
            "drop3"
            "header4"
            "toggle"
            ''')
       )
        display(controls)

        #Update plot
        self.target_labels = [self.available_targets[i] for i in self.targets]
        plot_main(targets=self.target_labels, 
                  base_target=self.selected_base_dropdown.value, 
                  ref_target=self.selected_reference_dropdown.value, 
                  center_toggle=not self.centered_toggle.value, 
                  filename_prefix=self.filename_prefix, 
                  x_max=self.x_max, 
                  electrodes=self.electrode)

    def change_and_refresh(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.refresh()
    
    def select_comp(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if self.selected_base_dropdown.value != ' ':
                self.base_target = self.selected_base_dropdown.value
            else:
                self.base_target = None
            if self.selected_reference_dropdown.value != ' ':
                self.ref_target = self.selected_reference_dropdown.value
            else:
                self.ref_target = None
            self.refresh()

    def change_dataset(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            dataset_index = self.datasets.index(self.select_dataset_dropdown.value)
            self.filename_prefix = self.filename_prefixes[dataset_index]
            self.x_max = self.x_maxes[dataset_index]
            self.electrode = self.electrodes[dataset_index]
            self.available_targets, _ = retrieveData(0, 1)
            self.refresh()

    def change_targets(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.targets = []
            for i, target_checkbox in enumerate(self.targets_checkboxes):
                if target_checkbox.value:
                    self.targets.append(i)
            self.target_labels = [self.available_targets[i] for i in self.targets]
            
            self.refresh()

###############################################
## FUNCTIONS                                 ##
###############################################

#Define function to determine filenames
def retrieveData(data, prefix='', electrodes=1): 
    #analysis_names = ['Empirical', 'GAN-Augmented', 'Oversampled-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']
    analysis_names = ['Empirical', 'GAN-Augmented', 'VAE-Augmented', 'Oversampled-Augmented', 'Gaussian-Augmented', 'Flip-Augmented', 'Reverse-Augmented', 'Smooth-Augmented']

    filenames= [
        f'{prefix}Classification Results/empPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/ganPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/vaePredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/overPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/gausPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/negPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/revPredictions_e{electrodes}_NN_TestClassification.csv',
        f'{prefix}Classification Results/smoothPredictions_e{electrodes}_NN_TestClassification.csv',
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
    x_axis_scale = [5, 10, 15, 20] if len(meanData) == 4 else [5, 10, 15, 20, 25, 30, 35]
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

    if target == ' ' and reference != ' ':
        target = reference
        reference = ' '
    
    if target != ' ':
        meanDataDSSyn_SynP100 = []
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

        if reference != ' ':
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
        x_range = np.arange(1,5) if len(meanDataDS) == 4 else np.arange(1,8)
        if reference != ' ':
            plt.bar(x_range,meanDiff,color='grey',width=.5)
        else:
            plt.bar(x_range,meanDataDS,color='grey',width=.5)
        
        for i in range(len(meanDataDS)):
            annotation_font_size=8
            if reference != ' ':
                if meanDiff[i] > 0:
                    plt.annotate(str(round(meanDiff[i]))+'%', (x_range[i],meanDiff[i]+.2), ha='center', color='grey', size = annotation_font_size)
                else:
                    plt.annotate(str(round(meanDiff[i]))+'%', (x_range[i],.2), ha='center', color='grey', size = annotation_font_size) 
            else:
                plt.annotate(str(round(meanDataDS[i]))+'%', (x_range[i],meanDataDS[i]+.5), ha='center', color='grey', size = annotation_font_size)
            
#Main plotting function
def plot_main(targets=[], base_target=' ', ref_target=' ', center_toggle=True, filename_prefix='', interactive=True, save_name=None, x_max = 20, electrodes=1):
    
    ###############################################
    ## SETUP                                     ##
    ###############################################

    #Determine the sample sizes of interest
    xLabels = [5,10,15,20] if x_max == 20 else [5,10,15,20,30,60,100]
    data = np.arange(1,6)
        
    ###############################################
    ## SETUP FIGURE                              ##
    ###############################################

    #Figure Parameters
    ylims = 80
    xlims = 22.5 if x_max == 20 else 37.5
    alpha = .6
    alpha_selected = .8
    alpha_targets = .1
    alpha_nontargets = .1
    fontsize = 10 if interactive else 14

    #Setup
    plot_length = 15 if interactive else 10
    fig = plt.figure(figsize=(plot_length, 15))
    plt.rcParams.update({'font.size': 5})  
    ylabel_x = 0.09 if interactive else 0.01
    fig.text(ylabel_x, 0.5, 'Prediction Accuracy (%)', va='center', rotation='vertical', fontsize=fontsize)
    plot_columns = 4 if interactive else 3

    ###############################################
    ## PLOT NEURAL NETWORK                       ##
    ###############################################

    #Iterate through all subplots
    for dat in data:
    
        #Signify subplot
        dat_index = dat-1
        ax1 = plt.subplot2grid((5, plot_columns), (dat_index, 0), colspan=3)

        #Create horizontal lines
        axisLevels = np.arange(50,ylims,5)
        for y in axisLevels:
            plt.axhline(y=y, color='k', linestyle=':', alpha=.05, label='_nolegend_')

        #Load and plot data while extracting legend names
        legendNames = []
        analysis_names, filenames = retrieveData(dat, filename_prefix, electrodes=electrodes)
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
        x_axis_scale = [5, 10, 15, 20] if x_max == 20 else [5, 10, 15, 20, 25, 30, 35]
        plt.ylim(45,ylims)
        plt.xlim(2.5, xlims)
        if interactive:
            plt.xticks(x_axis_scale, xLabels, fontsize=fontsize-2)
            plt.yticks(np.arange(50,ylims,5), fontsize=fontsize-2)
        else:
            plt.xticks(x_axis_scale, xLabels, fontsize=fontsize)
            plt.yticks(np.arange(50,ylims,5), fontsize=fontsize)

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
            legend_anchor = .25 if interactive else .3
            ncol= np.ceil(len(legendNames)/2) if interactive else np.ceil(len(legendNames)/3)
            legend_font_offset = 3 if interactive else 5
            plt.legend(legend_elements, legendNames, frameon=False, ncol=ncol, fontsize=fontsize-legend_font_offset, bbox_to_anchor=(legend_anchor,1))
            
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
        row_label_x = 2.75 if interactive else 2.75
        row_label_y = ylims if interactive else ylims-2
        if (dat == 1):
            ax1.annotate('Neural Network', (row_label_x,row_label_y), fontsize = fontsize)
        elif (dat == 2):
            ax1.annotate('Support Vector Machine', (row_label_x,row_label_y), fontsize = fontsize)
        elif (dat == 3):
            ax1.annotate('Logistic Regression', (row_label_x,row_label_y), fontsize = fontsize)
        elif (dat == 4):
            ax1.annotate('Random Forest', (row_label_x,row_label_y), fontsize = fontsize)
        elif (dat == 5):
            ax1.annotate('K-Nearest Neighbors', (row_label_x,row_label_y), fontsize = fontsize)
        
        #Add difference subplot if two+ targets are selected
        if interactive:
            ax1 = plt.subplot2grid((5, 4), (dat_index, 3), colspan=1)
            plotDiffData(ref_target, base_target, analysis_names, filenames)
            ax1.spines[['right', 'top']].set_visible(False)
            x_range = np.arange(1,5) if x_max == 20 else np.arange(1,8)
            x_ticks = ['5','10','15','20'] if x_max == 20 else ['5','10','15','20','30','60','100']
            plt.xlim(0,len(x_ticks)+1)
            plt.xticks(x_range, x_ticks, fontsize=fontsize-2)

            if base_target != ' ' and ref_target != ' ':
                plt.ylim(0,20)
                plt.yticks(np.arange(0,20,5), fontsize=fontsize-2)
            else:
                plt.ylim(50,ylims)
                plt.yticks(np.arange(50,ylims,5), fontsize=fontsize-2)

            if dat == 1:
                if base_target != ' ' and ref_target != ' ':
                    ax1.annotate(f"{base_target.replace('-Augmented','')} - {ref_target.replace('-Augmented','')}",(0.55,15), fontsize = fontsize)
                elif base_target != ' ':
                    ax1.annotate(f'{base_target}',(0.55,ylims), fontsize = fontsize)
                elif ref_target != ' ':
                    ax1.annotate(f'{ref_target}',(0.55,ylims), fontsize = fontsize)
                else:
                    ax1.annotate('Select a comparison',(0.55,ylims), fontsize = fontsize)
            elif dat == 5:
                plt.xlabel('Sample Size', fontsize=fontsize)


    ###############################################
    ## SAVE PLOT                                 ##
    ###############################################
    if not save_name:
        plt.show()
    else:    
        fig = plt.gcf()
        fig.set_size_inches(plot_length, 15)
        plt.tight_layout()
        plt.subplots_adjust(left=0.075)
        fig.savefig(save_name, dpi=600, facecolor='white', edgecolor='none')