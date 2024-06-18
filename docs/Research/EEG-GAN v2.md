---
hide:
    -toc
---

# Research

## <b>Enhancing EEG Data Classification Across Diverse Contexts Using Generative Adversarial Networks</b>
### <i>Williams, Weinhardt, Hewson, Plomecka, Langer, & Musslick (in prep)</i>

<center> [Journal Print](TBD){ .md-button} [Data and Scripts](TBD){ .md-button } </center>

#### Abstract: ####

Electroencephalography (EEG) is crucial for studying cognition and is widely used in neurotechnology applications. However, EEG data is challenging to interpret and utilize effectively in machine learning, which requires large datasets that are time-consuming and costly to collect. Recent generative artificial intelligence (AI) advancements offer a solution by creating realistic synthetic EEG samples to expand datasets, thereby improving classification performance. Foundational studies show the benefits of using generative AI in EEG-based machine learning, but they often focus on specific use cases, leaving the general robustness of these enhancements across various contexts to be determined. This study evaluated the application of a generative adversarial network (GAN) for producing realistic EEG samples and enhancing classification performance across four datasets, five classifiers, and seven sample sizes, comparing the results to six benchmark augmentation techniques. The augmentation led to performance gains of up to 16\%, with enhancements consistent across datasets but varying among classifiers. GAN augmentations were particularly effective for smaller sample sizes (30 and below), improving 90\% of classification analyses. GAN augmentation also surpassed the classification enhancements of six benchmark techniques 90\% of the time. These findings suggest that GANs can generate high-quality EEG data reliably, offering a cost-effective alternative to extensive data collection. Enhanced data quality from GANs can improve applications in brain-computer interfacing, educational training, and neurofeedback and has potential clinical implications for early diagnosis and treatment of neurological disorders.

<center> [Online Interactive Figure](EEG-GAN v2 interactive.ipynb){ .md-button } </center>

<center> ![](./Images/Figure 4 - GAN Classification Results.png){: style="height:800px;width:800px"}</center> 

## <b>Code and Script Availability</b>

### Script Repos

<center> 
[Reinforcement Learning](https://github.com/AutoResearch/EEG-GAN/tree/manuscript-reinforcement_learning_task){ .md-button} 
[Anti-Saccade](https://github.com/AutoResearch/EEG-GAN/tree/manuscript-antisaccade_task){ .md-button } 
[Face Perception & Visual Search](https://github.com/AutoResearch/EEG-GAN/tree/manuscript-ERPCORE_tasks){ .md-button} 
[Figures](https://github.com/AutoResearch/EEG-GAN/tree/manuscript-results){ .md-button} 
</center>

### Data and Models
<center> 
[Data & Classification Results](https://osf.io/mj9cz/){ .md-button} 
[Autoencoders](https://osf.io/znv7k/){ .md-button} 
[Generative Models and Data](https://osf.io/s4agq/){ .md-button} 
</center>