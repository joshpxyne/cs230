# Alphanumeric-Augmented-CNN

Augments the EMNIST dataset to recognize more deviant non-standard alphanumeric characters.

Input ->  3x3 kernel size convolutional layer with 32 filters and ReLU -> max-pooling layer with 2x2 pooling window -> flattening layer -> fully-connected 1024-neuron layer with ReLU -> 1/5 neuron dropout layer -> fully-connected 512-neuron layer with ReLU -> 1/5 neuron dropout layer -> output softmax)

Given bayes error, doing well at around 85% test accuracy after a few epochs

Tests on an image of your choice at the end...must be in the correct format (28x28, grayscale, white marking on black background). Code for formatting characters is in another repo.

Must add EMNIST dataset in /data, train for weights (comment in model training code)