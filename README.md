# Picnic Image Classifier
This repository contains code for the Picnic Hackathon: https://picnic.devpost.com/ 

The code contains an image classifier built using keras.

train.py is the file that trains the convolutional neural network. In order to train the network, this file should be run.

enhance_dataset.py contains code that adds extra augmented images to the dataset in order provide more training and testing data for the convolutional neural network.

move_files.py contains code that moves the images in the standard dataset provided by the Picnic team, into a dataset that works with keras's .flow_from_directory() method.
