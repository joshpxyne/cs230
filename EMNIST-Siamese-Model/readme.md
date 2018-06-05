# EMNIST-Siamese-CNN

Trains a Siamese Convolutional Neural Network on the Extended MNIST dataset.
Reaches ~97% accuracy for determining matches between characters in 5 epochs.

Uses two convolutional layers, a pooling and flattening layer, and a fully-connected layer (ReLU as activation where applicable). Uses the contrastive loss function commonlyused in triplet models, using Euclidean distance between encodings to predict matches. RMS Prop is used for the optimization function, and the other hyperparameters are Keras' defaults.

Based off work from github user mmmikael: https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d
