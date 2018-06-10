from __future__ import absolute_import
from __future__ import print_function

import os
from os.path import isfile

from keras.preprocessing import image as im

training = 5000 # rough estimate, will generate a number nondeterministically lower than 5000 of images

datagen = im.ImageDataGenerator(
    rotation_range=70,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    featurewise_center=True, 
    featurewise_std_normalization=True,
    fill_mode='nearest')

def createShapes(g, name):

  os.makedirs("training-images1/"+name)
  i = 0
  trip = False
  img = im.load_img("training-template/"+g)  # this is a PIL image
  x = im.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
  x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
  datagen.fit(x)
  for batch in datagen.flow(x, batch_size=1, save_to_dir="training-images1/"+name, save_prefix="im", save_format='png'):
    i += 1
    if i >= training: break

createShapes("Circle.jpg", "Circle")
createShapes("Diamond.jpg", "Diamond")
createShapes("Plus.jpg","Plus")
createShapes("QuarterCircle.jpg","QuarterCircle")
createShapes("Rectangle.jpg","Rectangle")
createShapes("SemiCircle.jpg","SemiCircle")
createShapes("Star.jpg","Star")
createShapes("Trapezoid.jpg","Trapezoid")
createShapes("Triangle.jpg","Triangle")

