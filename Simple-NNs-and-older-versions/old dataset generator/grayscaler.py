# this converts all images in given directories to grayscale. 
# Must already have images in training-images1 directories generated from reader.py in keras-image-preprocessing.

# usage: $ python grayscaler.py

import cv2
import sys
import os
from os.path import isfile

from keras.preprocessing import image as im

i = 180000

for f in os.listdir("training-images1"):
	if not isfile("training-images1/"+f):
		for g in os.listdir("training-images1/"+f):
			if not os.path.exists("conv-train/"+f):
				os.makedirs("conv-train/"+f)
			if g.endswith("png"):
				img = cv2.imread("training-images1/"+str(f)+"/"+str(g), 0)
				cv2.imwrite("conv-train/"+str(f)+"/im"+str(i)+".png", img)
				i = i+1

for f in os.listdir("test-images1"):
	if not isfile("test-images1/"+f):
		for g in os.listdir("test-images1/"+f):
			if not os.path.exists("conv-test/"+f):
				os.makedirs("conv-test/"+f)
			if g.endswith("png"):	
				img = cv2.imread("test-images1/"+str(f)+"/"+str(g),0)
				cv2.imwrite("conv-test/"+str(f)+"/im"+str(i)+".png", img)
				i = i+1