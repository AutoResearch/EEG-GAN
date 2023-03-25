---
hide:
    -toc
---

# Augmenting EEG with GANs

## <b>The GAN Package</b>

We here use Generative Adversarial Networks (GANs) to create trial-level synthetic EEG samples. We can then use these samples as extra data to train whichever classifier we want to use (e.g.,  Support Vector Machine, Neural Network).

GANs are machine learning frameworks that consist of two adversarial neural network agents, namely the generator and the discriminator. The generator is trained to create novel samples that are indiscernible from real samples. In the current context, the generator produces realistic continuous EEG activity, conditioned on a set of experimental variables, which contain underlying neural features representative of the outcomes being classified. For example, depression manifests as increased alpha oscillatory activity in the EEG signal, and thus, an ideal generator would produce continuous EEG that includes these alpha signatures. In contrast to the generator, the discriminator determines whether a given sample is real or synthetically produced by the generator. The core insight of GANs is that the generator can effectively learn from the discriminator. Specifically, the generator will consecutively produce more realistic synthetic samples with the goal of “fooling” the discriminator into believing them as real. Once it has achieved realistic samples that the discriminator cannot discern, it can be used to generate synthetic data—or in this context, synthetic EEG data.

## <b>Evaluation of the GAN Package</b>
<b>Augmenting EEG with Generative Adversarial Networks Enhances Brain Decoding Across Classifiers and Sample Sizes</b><br>
&emsp;*Williams, Weinhardt, Wirzberger, & Musslick (*submitted, 2023*)*<br>

## <b>About</b>

This project is in active development by the [Autonomous Empirical Research Group](https://musslick.github.io/AER_website/Research.html), led by [Sebastian Musslick](https://smusslick.com). The package was built by [Sebastian Musslick](https://smusslick.com), Daniel Weinhardt, and [Chad Williams](http://www.chadcwilliams.com).

This research program is supported by Schmidt Science Fellows, in partnership with the Rhodes Trust, as well as the Carney BRAINSTORM program at Brown University.


