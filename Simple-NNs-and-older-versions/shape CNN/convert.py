# converts png image in dataset to an numpy array

import numpy as np
import glob, os
from PIL import Image

numC=3981
numD=3928
numP=3905
numQC=3922
numR=3945
numSC=3983
numS=3969
numTra=3910
numTri=3977

width=50
height=50

predChar = np.zeros((numC+ # keras data generation is not reliable, so manual input of class batch size is necessary
numD+
numP+
numQC+
numR+
numSC+
numS+
numTra+
numTri,(width*height)+1))

def add(g,index,ind):
	j=0
	for file in os.listdir("training-images1/"+g):
	    if file.endswith(".png"):
			predChar[index+j][0] = ind
			im=Image.open("training-images1/"+g+"/"+file)
			img=im.convert("L")
			shape = np.array(img).reshape(width*height,1)
			for i in range(width*height):
				predChar[index + j][i+1] = shape[i][0]
			j+=1

add("Circle",0,0)
add("Diamond",numC,1)
add("Plus",numC+numD,2)
add("QuarterCircle",numC+numD+numP,3)
add("Rectangle",numC+numD+numP+numQC,4)
add("SemiCircle",numC+numD+numP+numQC+numR,5)
add("Star",numC+numD+numP+numQC+numR+numSC,6)
add("Trapezoid",numC+numD+numP+numQC+numR+numSC+numS,7)
add("Triangle",numC+numD+numP+numQC+numR+numSC+numS+numTra,8)

np.save("shapes",predChar)