---
hide:
    -toc
---
# GAN Package Main Functions

## <b>GAN Package Details</b>
<b>There are three main functions from the GANs package: </b><br>
&emsp;&emsp;```train_gan.py``` - This trains a GAN <br>
&emsp;&emsp;```visualize_gan.py``` - This trains a GAN <br>
&emsp;&emsp;```generate_samples.py``` - This generates synthetic samples using the trained GAN

<i>You can run these functions from terminal:</i><br>
&emsp;&emsp;```python train_gan.py```<br>

<i>You can also run these functions from your script:</i><br>
&emsp;&emsp;```train_gan(argv)```<br>

<b>Arguments:</b><br>

<i>In terminal, arguments are stated after the script filename:</i><br>
&emsp;&emsp;```python train_gan.py path_dataset=data\my_data.csv n_epochs=100```

<i>As a function, arguments are defined as a dictionary:<br>
&emsp;&emsp;```from train_gan import *```<br>

&emsp;&emsp;```argv = dict(```<br>
&emsp;&emsp;&emsp;```path_dataset=data\my_data.csv,```<br>
&emsp;&emsp;&emsp;```n_epochs = 100```<br>
&emsp;&emsp;```)```<br>

&emsp;&emsp;```train_gan(argv)```

<b>For the files in this package, you can use the help argument to see a list of possible arguments with a brief description:</b><br>
&emsp;&emsp;```python train_gan.py help```<br>
&emsp;&emsp;```train_gan(dict(help = True))```

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">GAN Training Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>python train_gan.py help</code><br>
&emsp;&emsp;<code>train_gan(dict(Help = True))</code><br>
<img src="../Images/GAN-Training-Help.png" alt=""><br>
<img src="../Images/GAN-Training-Help-2.png" alt=""><br>
<img src="../Images/GAN-Training-Help-3.png" alt=""><br>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Visualize Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>python visualize_gan.py help</code><br>
&emsp;&emsp;<code>visualize_gan(dict(Help = True))</code><br>
<img src="../Images/Visualize-Help.png" alt=""><br>
<img src="../Images/Visualize-Help-2.png" alt=""><br>
<img src="../Images/Visualize-Help-3.png" alt=""><br>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Generate Samples Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>python generate_samples.py help</code><br>
&emsp;&emsp;<code>generate_samples(dict(Help = True))</code><br>
<img src="../Images/Generate-Samples-Help.png" alt=""><br>
<img src="../Images/Generate-Samples-Help-2.png" alt=""><br>
<img src="../Images/Generate-Samples-Help-3.png" alt="">
</details>