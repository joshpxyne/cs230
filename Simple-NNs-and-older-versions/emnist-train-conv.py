### 7-hidden-layer Convolutional Neural Network ###
### Ballpark test accuracy after 1 epoch: 83% ###
### Ballpark time to complete one training epoch and test: 100s ###
# Reference: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import numpy as np
import pandas as pd
import random
import time

import keras as K

## SET UP DATA ##

train_db = pd.read_csv("data/emnist-balanced-train.csv").values # convert data to numpy array
test_db  = pd.read_csv("data/emnist-balanced-test.csv").values

charInd = random.randint(0,112000) # select random index in dataset for testing
emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

num_classes = 47
y_train = train_db[:,0]
y_train = K.utils.np_utils.to_categorical(y_train, num_classes)

x_train = train_db[:,1:]
x_train = np.reshape(x_train, (112799,28,28))
x_train = x_train.astype('float32')
x_train /= 255 # normalize training data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

alphanum = np.where(y_train[charInd]==1.)[0][0] # get index character in one-hot vector label

## SET UP MODEL ##

m = time.time()

model = K.models.Sequential()
model.add(K.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape)) # first layer is a convolutional layer as specified
# model.add(K.layers.Conv2D(64, (3, 3), activation='relu')) # Slows training significantly, not a huge boost in accuracy
model.add(K.layers.MaxPooling2D(pool_size=(2, 2))) # do max pooling after convolution
# model.add(K.layers.Dropout(0.25)) # may not need second dropout layer without extra convolution
model.add(K.layers.Flatten()) # flatten in order to connect FC layer
model.add(K.layers.Dense(1024, activation='relu'))# fully connected layer, relu, 1024 neurons
model.add(K.layers.Dropout(0.2)) # set 1/5 of parameters to zero
model.add(K.layers.Dense(512, activation='relu')) # Another FC layer
model.add(K.layers.Dropout(0.2)) # set 1/5 of parameters to zero
model.add(K.layers.Dense(num_classes, activation='softmax')) # softmax for output

model.compile(loss='categorical_crossentropy', # cross-entropy loss function, categorical because of softmax
              optimizer='adam', # Adam optimizer function
              metrics=['accuracy']) # report accuracy

model.fit(x_train, y_train, # train the model using the training set
          batch_size=512, epochs=5,
          verbose=1, validation_split=0.05) # carve out 5% data for validation

## TEST/EVALUATE ##

y_test = test_db[:,0]
y_test = K.utils.np_utils.to_categorical(y_test, num_classes)

x_test = test_db[:,1:]
x_test = x_test.astype('float32')
x_test /= 255 # normalize data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(model.evaluate(x_test, y_test, verbose=1)) # evaluate the trained model on the test set

## UTILIZE ##

randChar = np.array([x_train[charInd,:]])
prediction = model.predict(randChar)
print(prediction)

pred = 0
for i in range(47):
	if prediction[0][i] > pred:
		pred = prediction[0][i]
		print(prediction[0][i])
		predIndex = i

print("Randomly selected test character: "+str(emnist[alphanum]))
print("Model's prediction of this character: " + emnist[predIndex]+", with a probability of " + str(100*pred)+"%")
print("Time to train (s): " + str(time.time() - m))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")