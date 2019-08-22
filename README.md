# Dog Classifier

## Table of Contents

1. [Installation](#installation)
2. [Project Overview](#projectoverview)
3. [File Descriptions](#files)
4. [Results](#results)

## Installation

Beyond the Anaconda distribution of Python, the following packages need to be installed:

* opencv-python
* h5py
* matplotlib
* numpy
* scipy
* tqdm
* scikit-learn
* keras
* tensorflow==1.0.0

## Project Overview

As part of this project, I built and trained a neural network model using converged neural networks (CNN), using 8,351 photos of 133 dog breeds. CNN is a type of deep neural network commonly used to analyze image data. Typically, CNN architecture includes convolutional layers, activation function, cluster layers, fully connected layers, and normalization layers. Transfer learning is a technique that allows you to reuse a model developed for one task as a starting point for another task.

A trained model can be used by a web or mobile application to process realistic images provided by the user. When you give a picture of a dog, the algorithm predicts the breed of the dog. If a human image is provided, the code will identify the most similar dog breeds.

## File Descriptions

Below are main foleders/files for this project:

1. haarcascades

    * haarcascade_frontalface_alt.xml:  a pre-trained face detector provided by OpenCV

2. bottleneck_features

    * DogVGG19Data.npz: pre-computed the bottleneck features for VGG-19 using dog image data including training, validation, and test

3. saved_models

    * VGG19_model.json: model architecture saved in a json file
    * weights.best.VGG19.hdf5: saved model weights with best validation loss ( Transfer Learning VGG19 )
    * weights.best.VGG16.hdf5: saved model weights with best validation loss ( Transfer Learning VGG16 )
    * weights.best.Resnet50.hdf5: saved model weights with best validation loss ( Transfer Learning Resnet50 )
    * weights.best.from_scratch.hdf5: saved model weights with best validation loss ( custom CNN network )

4. custom_images: a few image files for test

5. dog_app.ipynb: a notebook used to build and train the dog breeds classification model

6. extract_bottleneck_features.py: functions to compute bottleneck features given a tensor converted from an image

7. images: a few images to test the model manually

## Results

1. The model was able to reach an accuracy of 72.9665% accuracy on test data using VGG19 model
2. If a dog image is supplied, the model gives a prediction of the dog breed.
3. The model is also able to identify the most resembling dog breed of a person

More discussions can be found in this blog: <url>
