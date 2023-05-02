---
hide:
    -toc
---
# Installing EEG-GAN

## <b>Pre-Requisites</b>

The only prerequisites for installing EEG-GAN is [Python](https://www.python.org/downloads/).

---

## <b>Install With Pip</b>

To install using [pip](https://pip.pypa.io/en/stable/cli/pip_download/):<br>

&emsp;&emsp;```pip install eeggan```

You can then import the package functions directly:

&emsp;&emsp;<i>Import functions:</i> ```from eeggan import train_gan, visualize_gan, generate_samples```<br>
&emsp;&emsp;<i>Use function:</i> ```train_gan(argv)```

Or indirectly: 

&emsp;&emsp;<i>Import package:</i> ```import eeggan as eg```<br>
&emsp;&emsp;<i>Use function:</i> ```eg.train_gan(argv)```

<i>If you prefer, follow this [link](https://pypi.org/project/eeggan/){.new_tab} to our PyPI page to directly download wheels or source.</i>

---

## <b>Installing From the Source</b>
You can install the latest GitHub version of EEG-GAN into your environment:<br>

&emsp;&emsp;```pip install git+https://github.com/AutoResearch/EEG-GAN@3-release-as-pip-package```

<i>Note that our pip release is currently on the branch 3-release-as-pip-package.</i><br>
<i>Note that the GitHub repo may be ahead of the latest pip release.</i><br>
<i>For most stable results, we suggest you follow the Install With Pip instructions above.</i>

---

## <b>Obtaining the Source</b>
If you rather download the package via GitHub you can use the following command:<br>

&emsp;&emsp;```git clone https://github.com/AutoResearch/EEG-GAN.git```

To update your local package:

&emsp;&emsp;```git pull```

<i>Note that the GitHub repo may be ahead of the latest pip release.</i> <br>
<i>For most stable results, we suggest you follow the Install With Pip instructions above.</i>

---

## <b>Dependencies</b>
The following are the required dependencies for EEG-GAN. If you install EEG-GAN via pip, or installed EEG-GAN from the source, these are automatically installed. If you have obtained the package from the source you can install these via the requirements.txt file ```pip install -r requirements.txt``` or via the pyproject.toml file ```pip install -e .[dev]```

&emsp;&emsp;[Pandas](https://pandas.pydata.org/)~=1.3.4<br>
&emsp;&emsp;[NumPy](https://numpy.org/)~=1.21.4<br>
&emsp;&emsp;[Matplotlib](https://matplotlib.org/)~=3.5.0<br>
&emsp;&emsp;[SciPy](https://scipy.org/)~=1.8.0<br>
&emsp;&emsp;[Torch](https://pytorch.org/)~=1.12.1<br>
&emsp;&emsp;[TorchVision](https://pytorch.org/)~=0.13.1<br>
&emsp;&emsp;[TorchSummary](https://pytorch.org/)~=1.5.1<br>
&emsp;&emsp;[TorchAudio](https://pytorch.org/)~=0.12.1<br>
&emsp;&emsp;[Einops](https://github.com/arogozhnikov/einops)~=0.4.1<br>
&emsp;&emsp;[Scikit-Learn](https://scikit-learn.org/)~=1.1.2<br>