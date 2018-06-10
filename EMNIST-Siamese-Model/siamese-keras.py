### Run on Colaboratory. ###

from google.colab import files
uploaded = files.upload()

### Run on Colaboratory. ###

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    # Positive and negative pair creation, generates half of each
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(47)]) - 1
    for d in range(47):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 47)
            dn = (d + inc) % 47
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network():
    seq = Sequential()
    seq.add(Conv2D(64, (5, 5),activation='relu',data_format='channels_last',input_shape=(input_dim,input_dim,1))) # convolutional layer
    seq.add(Conv2D(64, (3, 3),activation='relu',data_format='channels_last'))
    seq.add(MaxPooling2D((2, 2))) # typical max-pooling
    seq.add(Flatten()) # flatten for fully-connected layer
    seq.add(Dense(1024, activation='relu')) # fully-connected layer
    seq.add(Dropout(0.5)) # dropout for avoiding overfitting
    seq.add(Dense(1024, activation='relu')) # fully-connected layer
    seq.add(Dropout(0.5))
    seq.add(Dense(128, activation='relu')) # output fully-connected layer, encoding in 128-dimensional space
    seq = Sequential()
    return seq
    return seq


def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

raw_data = pd.read_csv("emnist-balanced-train.csv")

train, validate = train_test_split(raw_data, test_size=0.1)

X_train = train.values[:,1:]
y_train = train.values[:,0]

X_test = validate.values[:,1:]
y_test = validate.values[:,0]

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

input_dim = 28
nb_epoch = 5

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(47)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(47)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
base_network = create_base_network()

input_a = Input(shape=(input_dim,input_dim,1))
input_b = Input(shape=(input_dim,input_dim,1))

# because we re-use the same instance 'base_network',
# weights of the network will be shared across the two branches.
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          epochs=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))