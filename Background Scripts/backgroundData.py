#background generator by Peggy
#generates 50x50 pixels slices/croppings of background given a background image
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def sliceBackground(original):
    pixels = 50
    height, width, channels = original.shape
    sliceList = []
    for i in range(width-pixels):
        for j in range(height-pixels):
            slicedImg = original[i:i+pixels, j:j+pixels]
            sliceList.append(slicedImg)
            for rot in range(3):
                M = cv2.getRotationMatrix2D((pixels/2,pixels/2),90,1)
                slicedImg = cv2.warpAffine(slicedImg,M,(pixels,pixels))
                sliceList.append(slicedImg)

    return sliceList

def saveImageFromList(backgroundList):
    num = len(backgroundList)
    for i in range(num):
        cv2.imwrite("background" + str(i) + ".png", backgroundList[i])


img = cv2.imread("test.png")
backgrounds = sliceBackground(img)
saveImageFromList(backgrounds)
