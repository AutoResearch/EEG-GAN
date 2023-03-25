---
hide:
    -toc
---

# EEG-GAN Examples

## <b>GAN Training Examples</b>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Select Training Dataset</font></b></summary>
    <font size = "3">
    You can direct the GAN to train on specific datasets using the <code>path_dataset</code> argument. <br><br>

    &emsp;<code>python gan_training_main.py path_dataset=data\my_data.csv</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Number of Epochs</font></b></summary>
    <font size = "3">
    You can vary the number of epochs that the GAN is trained on with the <code>n_epochs</code> parameter. <br><br>

    &emsp;<code>python gan_training_main.py n_epochs=8000</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Continue Training a GAN</font></b></summary>
    <font size = "3">
    You can continue training a GAN using the <code>train_gan</code> and (optionally) <code>path_checkpoint</code> arguments. Not including the <code>path_checkpoint</code> argument will default to training a model <code>trained_models\checkpoint.pt</code> <br><br>

    &emsp;<code>python gan_training_main.py train_gan path_checkpoint=trained_models\my_model.pt</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Training on GPU</font></b></summary>
    <font size = "3">
    You can use your GPU rather than CPU to train the GAN using the <code>ddp</code> parameter.<br><br>

    &emsp;<code>python gan_training_main.py ddp</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Integrated GAN Training</font></b></summary>
    <font size = "3">
    We can use multiple arguments together to train our GAN, for example: <br>
    &emsp;On GPUs <code>ddp</code><br>
    &emsp;On our dataset <code>path_dataset=data\my_data.csv</code><br>
    &emsp;For 8000 epochs <code>n_epochs=8000</code><br><br>

    &emsp;<code>python gan_training_main.py ddp path_dataset=data\my_data.csv n_epochs=8000</code>
    </font>
</details>

## <b>Generate Samples Examples</b>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Select GAN Model</font></b></summary>
    <font size = "3">
    You can generate samples from a specific GAN using the <code>file</code> argument. <br><br>

    &emsp;<code>python generate_samples_main.py file=trained_models\my_GAN.pt</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Set Generated Samples Save Name</font></b></summary>
    <font size = "3">
    You can declare the path and name of the saved generated samples file using the <code>path_samples</code> argument. <br><br>

    &emsp;<code>python generate_samples_main.py path_samples=generated_samples\my_samples.csv</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Set Number of Samples to Generate</font></b></summary>
    <font size = "3">
    You can set the total number of samples to generate (which will be split equally across conditions) using the <code>num_samples_total</code> argument. <br><br>

    &emsp;<code>python generate_samples_main.py num_samples_total=10000</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Set Number of Samples to Generate in Parallel</font></b></summary>
    <font size = "3">
    You can set the number of samples that will be generated in parallel using the <code>num_samples_parallel</code> argument. <br><br>

    &emsp;<code>python generate_samples_main.py num_samples_parallel=1000</code>
    </font>
</details>

<details style="border-color:Grey;">
    <summary style="background-color:transparent;"><b><font size = "4">Integrated Generate Samples</font></b></summary>
    <font size = "3">
    We can use multiple arguments together to generate samples, for example: <br>
    &emsp;On our model <code>file=trained_models\my_GAN.pt</code><br>
    &emsp;With a saved filename  <code>path_samples=generated_samples\my_samples.csv</code><br>
    &emsp;Generating 10,000 samples <code>num_samples_total=10000</code><br>
    &emsp;At a rate of 1,000 at a time <code>num_samples_parallel=1000</code><br><br

    &emsp;<code>python generate_samples_main.py file=trained_models\my_GAN.pt path_samples=generated_samples\my_samples.csv num_samples_total=10000 num_samples_parallel=1000</code>
    </font>
</details>