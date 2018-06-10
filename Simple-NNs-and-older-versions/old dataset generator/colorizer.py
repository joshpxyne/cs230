# this converts all images in given directories to RGB. 
# Must already have images in training-template directories generated from reader.py in keras-image-preprocessing.

# usage: $ python colorizer.py

import cv2
import sys
import os
from os.path import isfile

from keras.preprocessing import image as im

i = 180000

for f in os.listdir("training-template"):
	if not isfile("training-template/"+f):
		for g in os.listdir("training-template/"+f):
			if not os.path.exists("conv-train/"+f):
				os.makedirs("conv-train/"+f)
			if g.endswith("png"):
				img = cv2.imread("training-template/"+str(f)+"/"+str(g),3)
				cv2.imwrite("conv-train/"+str(f)+"/im"+str(i)+".png", img)
				i = i+1

for f in os.listdir("test-template"):
	if not isfile("test-template/"+f):
		for g in os.listdir("test-template/"+f):
			if not os.path.exists("conv-test/"+f):
				os.makedirs("conv-test/"+f)
			if g.endswith("png"):	
				img = cv2.imread("test-template/"+str(f)+"/"+str(g),3)
				cv2.imwrite("conv-test/"+str(f)+"/im"+str(i)+".png", img)
				i = i+1