import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import random

def randomLocations(original, num, smallimg):
    height, width, channels = original.shape
    h, w, c = smallimg.shape
    locations = []
    for i in range(num):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        locations.append((x,y))

    for x, y in locations:
        #overlay images
        original[x:x+h, y:y+w] = smallimg
    cv2.imshow("test", original)
    cv2.waitKey(0)
    return locations


orig = cv2.imread("field.jpg")
overlay = cv2.imread("0.jpg")
print (randomLocations(orig, 2, overlay))
