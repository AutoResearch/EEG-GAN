---
hide:
    -toc
---

# Research

## <b>EEG-GAN: A Generative EEG Augmentation Toolkit for Enhancing Neural Classification.</b>
### <i>Williams, Weinhardt, Hewson, Plomecka, Langer, & Musslick (2025). bioRxiv</i>

<div style="text-align: center; margin-top: 1em;">
  <a href="TBD" class="md-button">
    Preprint
  </a>
</div>

#### Abstract: ####

Electroencephalography (EEG) is a widely applied method for decoding neural activity, offering insights into cognitive function and driving advancements in neurotechnology. However, decoding EEG data remains challenging, as classification algorithms typically require large datasets that are expensive and time-consuming to collect. Recent advances in generative artificial intelligence have enabled the creation of realistic synthetic EEG data, yet no method has consistently demonstrated that such synthetic data can lead to improvements in EEG decodability across diverse datasets. Here, we introduce EEG-GAN, an open-source generative adversarial network (GAN) designed to augment EEG data. In the most comprehensive evaluation study to date, we assessed its capacity to generate realistic EEG samples and enhance classification performance across four datasets, five classifiers, and seven sample sizes, while benchmarking it against six established augmentation techniques. We found that EEG-GAN, when trained to generate raw single-trial EEG signals, produced signals that reproduce grand-averaged waveforms and time-frequency patterns of the original data. Furthermore, training classifiers on additional synthetic data improved their ability to decode held-out empirical data. EEG-GAN achieved up to a 16% improvement in decoding accuracy, with enhancements consistent across datasets but varying among classifiers. Data augmentations were particularly effective for smaller sample sizes (30 and below), significantly improving 70% of these classification analyses and only significantly impairing 4% of analyses. Moreover, EEG-GAN significantly outperformed all benchmark techniques in 69% of the comparisons across datasets, classifiers, and sample sizes and was only significantly outperformed in 3% of comparisons. These findings establish EEG-GAN as a robust toolkit for generating realistic EEG data, which can effectively reduce the costs associated with real-world EEG data collection for neural decoding tasks.

![](./Images/Figure 4 - GAN Classification Results.png){: style="height:1200px;width:1200px"}

## <b>Script and Data Availability</b>

<h3>Script Repos</h3>
<div style="text-align: center; margin-bottom: 20px;">
  <a href="https://github.com/AutoResearch/EEG-GAN/tree/manuscript-reinforcement_learning_task" class="md-button">Reinforcement Learning</a>
  <a href="https://github.com/AutoResearch/EEG-GAN/tree/manuscript-antisaccade_task" class="md-button">Anti-Saccade</a>
  <a href="https://github.com/AutoResearch/EEG-GAN/tree/manuscript-ERPCORE_tasks" class="md-button">Face Perception & Visual Search</a>
  <a href="https://github.com/AutoResearch/EEG-GAN/tree/manuscript-results" class="md-button">Figures</a>
</div>

<h3>Data and Models</h3>
<div style="text-align: center;">
  <a href="https://osf.io/mj9cz/" class="md-button">Data & Classification Results</a>
  <a href="https://osf.io/znv7k/" class="md-button">Autoencoders</a>
  <a href="https://osf.io/s4agq/" class="md-button">Generative Models and Data</a>
</div>
