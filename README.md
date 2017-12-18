# Final Project for ECBM 4040 - Columbia University
#### Aarshay Jain, Jared Samet, Alex Wainger {aj2713, jss2272, atw2131}@columbia.edu

# Spectral Representations for Convolutional Neural Networks
This repo is an implementation of Rippel, Snoek, and Adams 2015 (https://arxiv.org/pdf/1506.03767.pdf). The project contains python modules and notebooks to implement the three proposals of the paper and to replicate the key findings and experiments of the paper.

## Final Report

A copy of our [Final Report](latex/final-report.pdf) is included in this repo.

## Requirements

The project was developed using Tensorflow 1.3.0 and NumPy 1.13. Certain notebooks require the Pillow 4.3 library to be installed (```sudo pip3 install Pillow```).

Since the code uses the NCHW format to perform convolutions, it will only run on a GPU-enabled machine.

The CIFAR-100 dataset does not come in batches. Loading the dataset will require a machine with at least 32 GB of RAM.

## Running saved models

Two of the notebooks refer to the saved models which contain the weights for our best accuracies. Before running these, unzip the ```src/best_model_10.tar.gz``` and ```src/best_model_100.tar.gz``` files.

## Code Organization

All code is located in the ```src``` folder. Within that folder, Python functions and classes that are shared between multiple notebooks are all located in the ```modules``` folder.

## Notebooks

[```approximation-loss.ipynb```](src/approximation-loss.ipynb) - This notebook demonstrates spectral pooling and frequency dropout in action on a minibatch. It also replicates the results for the approximation loss from the original paper.

[```cnn_spectral_parameterization.ipynb```](src/cnn_spectral_parameterization.ipynb) - Alex, please add a description

[```figure2.ipynb```](src/figure2.ipynb) - This notebook uses the spectral_pool function to downsample an image.

[```hyperparameter-search.ipynb```](src/hyperparameter-search.ipynb) - This notebook performs hyperparameter search on the CIFAR-10 dataset to identify the best hyperparameters.

[```full-training-10.ipynb```](src/full-training-10.ipynb) - This notebook uses the best identified hyperparameters to train the network on the entire CIFAR-10 dataset and compute the test accuracy. It also shows the improved results we got from manual tuning of the hyperparameters. Before running this notebook, unzip ```src/best_model_10.tar.gz```.

[```full-training-100.ipynb```](src/full-training-100.ipynb) - This notebook uses the best identified hyperparameters to train the network on the entire CIFAR-100 dataset and compute the test accuracy. It also shows the improved results we got from manual tuning of the hyperparameters. Before running this notebook, unzip ```src/best_model_100.tar.gz```.

## Modules

[```cnn_with_spectral_parameterization.py```](src/modules/cnn_with_spectral_parameterization.py) - This class builds and trains the generic and deep CNN architectures as described in section 5.2 of the paper with and without spectral parameterization of the filter weights. It was adapted from the homework assignment for this class on CNNs.

[```cnn_with_spectral_pooling.py```](src/modules/cnn_with_spectral_pooling.py) - This class builds the spectral pooling CNN architectures as described in section 5.1 of the paper. It was adapted from the homework assignment for this class on CNNs.

[```create_images.py```](src/modules/cnn_with_spectral_pooling.py) - These functions allowed us to experiment with and understand the behavior of the Fourier transform as applied to sample images.

[```frequency_dropout.py```](src/modules/frequency_dropout.py) - These functions implement the frequency dropout operation by creating a dropout mask with an input tensor for the truncation frequency.

[```image_generator.py```](src/modules/image_generator.py) - This class creates a data generator that supports image augmentation. It was adapted from the homework assignment for this class on CNNs.

[```layers.py```](src/modules/layers.py) - The classes in this file implement the various layers that we use to create the CNN architectures described in the paper. Some layers were adapted from the homework assignment on CNNs. The layers defined in this file are:
* ```default_conv_layer```: A standard convolutional layer with traditionally-parameterized weights
* ```fc_layer```: A fully connected dense layer
* ```spectral_pool_layer```: A layer implementing spectral pooling and frequency dropout
* ```spectral_conv_layer```: A convolutional layer with spectrally-parameterized weights
* ```global_average_layer```: A layer implementing global averaging as described in [Lin et al.](https://arxiv.org/abs/1312.4400)

[```spectral_pool.py```](src/modules/spectral_pool.py) - A function implementing spectral pooling that is shared by multiple sources

[```spectral_pool_test.py```](src/modules/spectral_pool_test.py) - Tests the spectral_pool function. This looks old and we should probably delete it.

[```utils.py```](src/modules/utils.py) - Various utility functions. Some were adapted from the homework assignment on CNNs.
