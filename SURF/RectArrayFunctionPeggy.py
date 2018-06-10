#returns array of rectangles of bounding box based on SURF keypoints given image by Peggy
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

class Rectangle:
	x = 0
	y = 0
	length = 0
	width = 0

# Function that returns an array of rectangles (upper left x, upper right y, length, width) of interesting points

def rectArrayReturn(keypointsList):
    rectArray = []
    # for every keypoint in the image, convert to rectangle
    for kp in keypointsList:
    	coordinate = kp.pt
    	rect = Rectangle()
    	rect.x = coordinate[0] - kp.size/2
    	rect.y = coordinate[1] - kp.size/2
    	rect.length = kp.size
    	rect.width = kp.size
    	rectArray.append(rect)
    #print (rectArray)
    return rectArray;
