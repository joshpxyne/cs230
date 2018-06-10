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

raw_data = np.load("shapes.npy")
shapes = ["Circle","Diamond","Plus","QuarterCircle","Rectangle","SemiCircle","Star","Trapezoid","Triangle"]

train, validate = train_test_split(raw_data, test_size=0.1)

x_train = train[:,1:]
y_train = train[:,0]


x_validate = validate[:,1:]
y_validate = validate[:,0]

batch_size = 512
num_classes = 9
epochs = 1

charInd = random.randint(0,3552) # select random index in dataset for testing

img_rows, img_cols = 50, 50

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
x_train /= 255 # normalize data
x_validate /= 255
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
model.add(Dense(1024, activation='relu')) # our usual two FC layers
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
 
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5,
                              patience = 2, min_lr = 0.0001)

# transfer learning, comment out if no model exists with this name in this directory
model.load_weights('shape-conv-model.h5')

model.fit_generator(datagen.flow(x_train, 
                                  y_train, 
                                  batch_size = batch_size), 
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (x_validate, y_validate),
                    callbacks = [reduce_lr])

score = model.evaluate(x_validate, y_validate, verbose = 0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# serialize model to JSON
model_json = model.to_json()

with open("shape-conv-model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("shape-conv-model.h5")
print("Saved model to disk")

randChar = np.array([x_validate[charInd,:]])

### convert to image format ###
ch = np.zeros((img_rows,img_cols))
for i in range(img_rows):
  for j in range(img_cols):
    ch[i][j] = randChar[0][i][j][0]
oldCh = ch
randDeg = random.randint(-45,45)
ch = imutils.rotate(ch,randDeg)
### convert back to model format
predChar = np.zeros((1,img_rows,img_cols,1))
for i in range(img_rows):
  for j in range(img_cols):
    predChar[0][i][j][0] = ch[i][j]

prediction = model.predict(predChar)
print(prediction)
pred = secondPred = predIndex = secondPredIndex = thirdPred = thirdPredIndex = 0

for i in range(num_classes):
  if prediction[0][i] > pred:
    pred = prediction[0][i]
    predIndex = i
for i in range(num_classes):
  if prediction[0][i] > secondPred and i != predIndex:
    secondPred = prediction[0][i]
    secondPredIndex = i
for i in range(num_classes):
  if prediction[0][i] > thirdPred and i != predIndex and i != secondPredIndex:
    thirdPred = prediction[0][i]
    thirdPredIndex = i


print("Random character: "+str(shapes[alphanum]))
print("1st guess: " + shapes[predIndex]+", probability: " + str(100*pred)+"%")
print("2nd guess: " + shapes[secondPredIndex]+", probability: " + str(100*secondPred)+"%")
print("3rd guess: " + shapes[thirdPredIndex]+", probability: " + str(100*thirdPred)+"%")

np.reshape(randChar, (img_rows,img_cols))
cv2.imshow("random character",np.array(ch))
cv2.waitKey(0)
