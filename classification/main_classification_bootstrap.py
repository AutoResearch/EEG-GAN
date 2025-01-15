###############################################
## LOAD MODULES                              ##
###############################################
import os
import pandas as pd
import numpy as np

###############################################
## FUNCTIONS                                 ##
###############################################

def bootstrap(data, n_iterations=10000):

    gan_data, bench_data = data
    data_means = [float(np.mean(gan_data)), float(np.mean(bench_data))]
    data_mean_diff = data_means[0] - data_means[1]
    
    gan_indices = np.random.randint(0, len(gan_data), size=(n_iterations, len(gan_data)))
    bench_indices = np.random.randint(0, len(bench_data), size=(n_iterations, len(bench_data)))
    gan_means = np.mean(gan_data[gan_indices], axis=1)
    bench_means = np.mean(bench_data[bench_indices], axis=1)
    distribution = (gan_means - bench_means) - data_mean_diff
    
    p_value = np.mean(np.abs(distribution) >= np.abs(data_mean_diff))
    sign = '+' if data_mean_diff > 0 else '-'
    sign = '' if p_value > 0.05 else sign
    p_value = f"{sign}{p_value:.4f}"

    return data_mean_diff, p_value

def significant_format(x):

    if x == '':
        return x
    
    format_color = 'LimeGreen' if x.startswith('+') else 'Maroon'
    x = x.replace('+', '').replace('-', '')
    if float(x) < 0.0001:
        formatted_p = f'\\textcolor{{{format_color}}}{{<.0001}}'
    elif float(x) < 0.05:
        #remove leading zeros
        x = x[1:] if x.startswith('0') else x
        formatted_p = f'\\textcolor{{{format_color}}}{{{x}}}'
    else:
        x = x[1:] if x.startswith('0') else x
        formatted_p = f'\\textcolor{{Gray}}{{{x}}}'

    return formatted_p

def extract_significance(data, columns, benchmark=None):
    if benchmark is not None:
        data = data[data['Benchmark'] == benchmark]
    bootstrap_results = data[columns]
    bootstrap_results = bootstrap_results.melt(value_vars=columns)['value'].values
    bootstrap_results = [x for x in bootstrap_results if x != '-']
    bootstrap_positive = [x for x in bootstrap_results if x.startswith('+')]
    bootstrap_negative = [x for x in bootstrap_results if x.startswith('-')]
    
    positive_results = len(bootstrap_positive) / len(bootstrap_results) * 100
    negative_results = len(bootstrap_negative) / len(bootstrap_results) * 100

    return [positive_results, negative_results]

###############################################
## RUN BOOTSTRAP                             ##
###############################################

def main():

    datasets = ['Reinforcement Learning', 'Antisaccade', 'ERPCORE/N170', 'ERPCORE/N2PC']
    classifiers = ['NN', 'SVM', 'LR', 'RF', 'KNN']
    benchmarks = ['emp', 'vae', 'over', 'gaus', 'rev', 'neg', 'smooth']
    
    bootstrap_dataframe = pd.DataFrame(columns=['Dataset', 'Classifier', 'Benchmark', '005', '010', '015', '020', '030', '060', '100'])
    for dataset in datasets:
        electrodes = ['e1', 'e8'] if dataset == 'Reinforcement Learning' else ['']
        for electrode in electrodes:
            for ci, classifier in enumerate(classifiers):
                if classifier != 'NN':
                    dataset_name = dataset if electrode == '' else f'{dataset} ({electrode})'
                    dataset_name = dataset_name.replace('Reinforcement Learning', 'RL').replace('Antisaccade', 'AS').replace('ERPCORE/N170', 'FP').replace('ERPCORE/N2PC', 'VS')
                    blank_dict = {'Dataset': dataset_name, 'Classifier': '', 'Benchmark': '', '005': '', '010': '', '015': '', '020': '', '030': '', '060': '', '100': ''}
                    bootstrap_dataframe = pd.concat([bootstrap_dataframe, pd.DataFrame([blank_dict])], ignore_index=True)
            
                #Find all files that contains the classifier in filename
                if dataset == 'Reinforcement Learning':
                    files = [f for f in os.listdir(f'classification/classification_results/{dataset}/Classification Results/') if f'_{classifier}_' in f and f'_{electrode}_' in f]
                else:
                    files = [f for f in os.listdir(f'classification/classification_results/{dataset}/Classification Results/') if f'_{classifier}_' in f]

                #Load file that begins with gan
                gan_file = [f for f in files if f.startswith('gan')][0]
                gan_file = f'classification/classification_results/{dataset}/Classification Results/{gan_file}'
                with open(gan_file, 'r') as f:
                    gan = f.readlines()
                gan = pd.DataFrame([x.split(',') for x in gan])
                gan = gan.iloc[:, [0, 3]]
                gan.columns = ['Sample Size', 'Performance']
                gan['Sample Size'] = pd.Categorical(gan['Sample Size'], categories=['005', '010', '015', '020', '030', '060', '100'], ordered=True)

                for benchmark in benchmarks:
                    print(f'{dataset} - {classifier} - {benchmark}')

                    bench_file = [f for f in files if f.startswith(benchmark)][0]
                    bench_file = f'classification/classification_results/{dataset}/Classification Results/{bench_file}'
                    with open(bench_file, 'r') as f:
                        bench = f.readlines()
                    bench = pd.DataFrame([x.split(',') for x in bench])
                    bench = bench.iloc[:, [0, 3]]
                    bench.columns = ['Sample Size', 'Performance']


                    comparison_mean_diff = []
                    comparison_p_value = []
                    for sample_size in gan['Sample Size'].unique():

                        gan_performance = gan[gan['Sample Size'] == sample_size]['Performance'].values.astype(int)
                        bench_performance = bench[bench['Sample Size'] == sample_size]['Performance'].values.astype(int)
                        performances = [gan_performance, bench_performance]

                        #Bootstrap
                        mean_diff, p_value = bootstrap(performances)

                        #Append to dataframe
                        comparison_mean_diff.append(mean_diff)
                        comparison_p_value.append(p_value)

                    results = {}
                    for ss, diff, p_val in zip(gan['Sample Size'].unique(), comparison_mean_diff, comparison_p_value):
                        results[ss] = p_val
                    if gan['Sample Size'].nunique() < 7:
                        results['030'] = '-'
                        results['060'] = '-'
                        results['100'] = '-'

                    if benchmark != 'emp':
                        classifier_name = ''
                    else:
                        classifier_name = classifier

                    dataset_name = dataset if electrode == '' else f'{dataset} ({electrode})'
                    dataset_name = dataset_name.replace('Reinforcement Learning', 'RL').replace('Antisaccade', 'AS').replace('ERPCORE/N170', 'FP').replace('ERPCORE/N2PC', 'VS')

                    comparison_dict = {'Dataset': dataset_name, 'Classifier': classifier_name, 'Benchmark': benchmark}
                    comparison_dict.update(results)
                    comparison_df = pd.DataFrame([comparison_dict])

                    bootstrap_dataframe = pd.concat([bootstrap_dataframe, comparison_df], ignore_index=True)
    
    #Determine % of significant results for GAN improvement
    empirical_sig_pos_reduced, empirical_sig_neg_reduced = extract_significance(bootstrap_dataframe, columns=['005', '010', '015', '020', '030'], benchmark='emp')
    all_sig_pos_reduced, all_sig_neg_reduced = extract_significance(bootstrap_dataframe, columns=['005', '010', '015', '020', '030'])
    empirical_sig_pos, empirical_sig_neg = extract_significance(bootstrap_dataframe, columns=['005', '010', '015', '020', '030', '060', '100'], benchmark='emp')
    all_sig_pos, all_sig_neg = extract_significance(bootstrap_dataframe, columns=['005', '010', '015', '020', '030', '060', '100'])
    significance_30andbelow = pd.DataFrame({'Empirical': [empirical_sig_pos_reduced, empirical_sig_neg_reduced], 'Benchmark': [all_sig_pos_reduced, all_sig_neg_reduced]})
    significance_all = pd.DataFrame({'Empirical': [empirical_sig_pos, empirical_sig_neg], 'Benchmark': [all_sig_pos, all_sig_neg]})
    significance_30andbelow.index = ['Positive', 'Negative']
    significance_all.index = ['Positive', 'Negative']

    bootstrap_dataframe = bootstrap_dataframe.replace('-', '999')
    for column in ['005', '010', '015', '020', '030', '060', '100']:
        bootstrap_dataframe[column] = bootstrap_dataframe[column].apply(lambda x: significant_format(x))
    bootstrap_dataframe = bootstrap_dataframe.replace('999', '-')
    bootstrap_dataframe['Benchmark'] = bootstrap_dataframe['Benchmark'].replace('emp', '\\textbf{Empirical}').replace('vae', '\\textbf{VAE}').replace('gaus', '\\textbf{Gaussian}').replace('neg', '\\textbf{Polarity Reverse}').replace('over', '\\textbf{Oversampling}').replace('rev', '\\textbf{Time Reverse}').replace('smooth', '\\textbf{Smoothing}')
    bootstrap_dataframe['Classifier'] = bootstrap_dataframe['Classifier'].replace('NN', '\\textbf{NN}').replace('SVM', '\\textbf{SVM}').replace('LR', '\\textbf{LR}').replace('RF', '\\textbf{RF}').replace('KNN', '\\textbf{KNN}')
    bootstrap_dataframe.columns = ['\\textbf{Dataset}', '\\textbf{Classifier}', '\\textbf{Comparison}', '\\textbf{SS 5}', '\\textbf{SS 10}', '\\textbf{SS 15}', '\\textbf{SS 20}', '\\textbf{SS 30}', '\\textbf{SS 60}', '\\textbf{SS 100}']

    #Save bootstrap_dataframe to latex
    rl1_df = bootstrap_dataframe[bootstrap_dataframe['\\textbf{Dataset}'].str.contains('e1')]
    rl1_df = rl1_df.drop(columns=['\\textbf{Dataset}'])
    rl8_df = bootstrap_dataframe[bootstrap_dataframe['\\textbf{Dataset}'].str.contains('e8')]
    rl8_df = rl8_df.drop(columns=['\\textbf{Dataset}'])
    as_df = bootstrap_dataframe[bootstrap_dataframe['\\textbf{Dataset}'].str.contains('AS')]
    as_df = as_df.drop(columns=['\\textbf{Dataset}'])
    fp_df = bootstrap_dataframe[bootstrap_dataframe['\\textbf{Dataset}'].str.contains('FP')]
    fp_df = fp_df.drop(columns=['\\textbf{Dataset}', '\\textbf{SS 30}', '\\textbf{SS 60}', '\\textbf{SS 100}'])
    vs_df = bootstrap_dataframe[bootstrap_dataframe['\\textbf{Dataset}'].str.contains('VS')]
    vs_df = vs_df.drop(columns=['\\textbf{Dataset}', '\\textbf{SS 30}', '\\textbf{SS 60}', '\\textbf{SS 100}'])

    #Save to latex
    caption = """Bootstrap results of the _DATASET_ dataset for the comparison of GAN performance with empirical and benchmark performance across classifiers and sample sizes.
                Values presented are the p-values of the mean difference.
                Green values indicate a significant increase in performance for the GAN relative to the corresponding comparison dataset,
                gray values indicate a non-significant difference in performance between the GAN and the comparison dataset,
                and red values indicate a significant decrease in performance for the GAN relative to the corresponding comparison dataset.
                SS = Sample Size, NN = Neural Network, SVM = Support Vector Machine, LR = Logistic Regression, RF = Random Forest, KNN = K-Nearest Neighbors."""
    rl1_df.to_latex('classification/classification_results/rl1_bootstrap_results.tex', index=False, caption=caption.replace('_DATASET_', 'Reinforcement Learning (e1)'), label='tab-S0A', column_format='llccccccc')
    rl8_df.to_latex('classification/classification_results/rl8_bootstrap_results.tex', index=False, caption=caption.replace('_DATASET_', 'Reinforcement Learning (e8)'), label='tab-S0B', column_format='llccccccc')
    as_df.to_latex('classification/classification_results/as_bootstrap_results.tex', index=False, caption=caption.replace('_DATASET_', 'Anti-Saccade'), label='tab-S0C', column_format='llccccccc')
    fp_df.to_latex('classification/classification_results/fp_bootstrap_results.tex', index=False, caption=caption.replace('_DATASET_', 'Face Processing'), label='tab-S0D', column_format='llccccc')
    vs_df.to_latex('classification/classification_results/vs_bootstrap_results.tex', index=False, caption=caption.replace('_DATASET_', 'Visual Search'), label='tab-S0E', column_format='llccccc')
    
    print('Bootstrap completed')

if __name__ == '__main__':
    np.random.seed(42)
    main()




                        

                        
                        

    



                
