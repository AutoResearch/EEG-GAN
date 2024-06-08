# GAN Package Main Functions

## <b>GAN Package Details</b>

<br><b>Functions</b><br>
There are three main functions from the EEG-GAN package:<br>
&emsp;&emsp;```train_gan()``` - This trains a GAN <br>
&emsp;&emsp;```visualize_gan()``` - This visualizes components of a trained GAN, such as the training losses <br>
&emsp;&emsp;```generate_samples()``` - This generates synthetic samples using the trained GAN<br>

<br><b>Arguments</b><br>

Each function can take a single argument ```argv```, which should be a dictionary:<br>

&emsp;&emsp;```argv = dict(```<br>
&emsp;&emsp;&emsp;```path_dataset=data\my_data.csv,```<br>
&emsp;&emsp;&emsp;```n_epochs = 100```<br>
&emsp;&emsp;```)```<br>

&emsp;&emsp;```train_gan(argv)```

<br><b>Help</b><br>

You can use the help argument to see a list of possible arguments with a brief description:</b><br>
&emsp;&emsp;```train_gan(dict(help = True))```<br>
&emsp;&emsp;```visualize_gan(dict(help = True))```<br>
&emsp;&emsp;```generate_samples(dict(help = True))```<br>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">GAN Training Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>train_gan(dict(Help = True))</code><br>
<img src="../Images/GAN-Training-Help.png" alt=""><br>
<img src="../Images/GAN-Training-Help-2.png" alt=""><br>
<img src="../Images/GAN-Training-Help-3.png" alt=""><br>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Visualize Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>visualize_gan(dict(Help = True))</code><br>
<img src="../Images/Visualize-Help.png" alt=""><br>
<img src="../Images/Visualize-Help-2.png" alt=""><br>
<img src="../Images/Visualize-Help-3.png" alt=""><br>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Generate Samples Help</font></b></summary>
    <font size = "3">
&emsp;&emsp;<code>generate_samples(dict(Help = True))</code><br>
<img src="../Images/Generate-Samples-Help.png" alt=""><br>
<img src="../Images/Generate-Samples-Help-2.png" alt=""><br>
<img src="../Images/Generate-Samples-Help-3.png" alt="">
</details>