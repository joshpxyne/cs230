# Generates images based of parameters set in the Keras ImageDataGenerator.

# usage: $ python generator.py

from __future__ import absolute_import
from __future__ import print_function

import os
from os.path import isfile

from keras.preprocessing import image as im

training = 100
testing = 25

datagen = im.ImageDataGenerator(
    rotation_range=70,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

for f in os.listdir("training-template"):
	if not isfile("training-template/"+f):
		if not os.path.exists("training-images1/"+f):
			os.makedirs("training-images1/"+f)
		for g in os.listdir("training-template/"+f):
			if g.endswith("png"):
				i = 0
				trip = False
				img = im.load_img("training-template/"+f+"/"+g)  # this is a PIL image
				x = im.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
				x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

				# the .flow() command below generates batches of randomly transformed images
				# and saves the results

				for batch in datagen.flow(x, batch_size=1, save_to_dir="training-images1/"+f, save_prefix="im", save_format='png'):
					i += 1
					
					if i >= training:
						path, dirs, files = os.walk("training-images1/"+f).next()
						gm = len(files)
						if gm == training or gm == training*2:
							trip = True
					if (trip):
						print(str(gm)+" images added to training-images1/"+f)
						break

for f in os.listdir("test-template"):
	if not isfile("test-template/"+f):
		if not os.path.exists("test-images1/"+f):
			os.makedirs("test-images1/"+f)
		for g in os.listdir("test-template/"+f):
			if g.endswith("png"):
				i = 0
				trip = False
				img = im.load_img("test-template/"+f+"/"+g)  # this is a PIL image
				x = im.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
				x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

				# the .flow() command below generates batches of randomly transformed images
				# and saves the results

				for batch in datagen.flow(x, batch_size=1, save_to_dir="test-images1/"+f, save_prefix="im", save_format='png'):
					i += 1
					if i >= testing:
						path, dirs, files = os.walk("test-images1/"+f).next()
						gm = len(files)
						if gm == testing or gm == testing*2:
							trip = True
					if (trip):
						print(str(gm)+" images added to test-images1/"+f)
						break


