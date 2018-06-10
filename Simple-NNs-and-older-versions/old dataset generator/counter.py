# usage: $ python counter.py

import os
from os.path import isfile


i = 0
j = 0

for f in os.listdir("conv-train"):
	if not isfile("conv-train/"+f):
		for g in os.listdir("conv-train/"+f):
			if g.endswith("png"):
				i = i+1

for f in os.listdir("conv-test"):
	if not isfile("conv-test/"+f):
		for g in os.listdir("conv-test/"+f):
			if g.endswith("png"):
				j = j+1

print("train: "+str(i))
print("test: "+str(j))
