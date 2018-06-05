from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as im
import cv2
import random
import numpy as np
import imutils

raw_data = pd.read_csv("data/emnist-balanced-train.csv")

train, validate = train_test_split(raw_data, test_size=0.1)

x_train = train.values[:,1:]
y_train = train.values[:,0]

x_validate = validate.values[:,1:]
y_validate = validate.values[:,0]

batch_size = 512
num_classes = 47
epochs = 1

charInd = random.randint(0,10000) # select random index in dataset for testing
emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_validate = x_validate.reshape(x_validate.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_validate = x_validate.astype('float32')
x_train /= 255
x_validate /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_validate.shape[0], 'validation samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_validate = keras.utils.to_categorical(y_validate, num_classes)

alphanum = np.where(y_validate[charInd]==1.)[0][0] # get index character in one-hot vector label


# Use data augmentation features of Keras
datagen = ImageDataGenerator(
    width_shift_range = 0.075,
    height_shift_range = 0.075,
    rotation_range = 45,
    shear_range = 0.075,
    zoom_range = 0.05,
    fill_mode = 'constant',
    cval = 0,
    
)

# datagen = ImageDataGenerator(zca_whitening=True)

datagen.fit(x_train)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape)) # Convolutional layer - 32 filters, 5x5 kernel size
# model.add(Conv2D(32, kernel_size=(3, 3), # Convolutional layer - 32 filters, 3x3 kernel size
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2))) # max pooling layer - 2x2 pool window size
# model.add(Dropout(0.25)) # dropout layer - sets 1/4 of the neurons to zero
          
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
 
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5,
                              patience = 2, min_lr = 0.0001)

# transfer learning
model.load_weights('conv-model.h5')

### comment back in to train ###
# model.fit_generator(datagen.flow(x_train, 
#                                   y_train, 
#                                   batch_size = batch_size), 
#                     epochs = epochs,
#                     verbose = 1,
#                     validation_data = (x_validate, y_validate),
#                     callbacks = [reduce_lr])

score = model.evaluate(x_validate, y_validate, verbose = 0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()

with open("conv-model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("conv-model.h5")
print("Saved model to disk")

randChar = np.array([x_validate[charInd,:]])

### convert to dataset format ###

newChar = cv2.imread("test3.png", 0)
newChar = np.array(newChar)/255.
newChar.reshape(28,28)
nChar = np.zeros((1,28,28,1))
for i in range(28):
  for j in range(28):
    nChar[0][i][j][0] = newChar.T[i][j]


prediction2 = model.predict(nChar)
print(prediction2)

pred = secondPred = predIndex = secondPredIndex = thirdPred = thirdPredIndex = 0

for i in range(47):
  if prediction2[0][i] > pred:
    pred = prediction2[0][i]
    predIndex = i
for i in range(47):
  if prediction2[0][i] > secondPred and i != predIndex:
    secondPred = prediction2[0][i]
    secondPredIndex = i
for i in range(47):
  if prediction2[0][i] > thirdPred and i != predIndex and i != secondPredIndex:
    thirdPred = prediction2[0][i]
    thirdPredIndex = i


print("Random character: "+str(emnist[alphanum]))
print("1st guess: " + emnist[predIndex]+", probability: " + str(100*pred)+"%")
print("2nd guess: " + emnist[secondPredIndex]+", probability: " + str(100*secondPred)+"%")
print("3rd guess: " + emnist[thirdPredIndex]+", probability: " + str(100*thirdPred)+"%")

np.reshape(nChar, (28,28))
cv2.imshow("random character",np.array(newChar))
cv2.waitKey(0)
