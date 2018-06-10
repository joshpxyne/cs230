#draws rectangular bounding bounding boxes from SURF and RectArrayFunction that Peggy made
#by Peggy

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from RectArrayFunctionPeggy import rectArrayReturn

img = cv2.imread('field1.jpg')
img[:, :, 0] = img[:, :, 2]
img[:, :, 1] = img[:, :, 2]
cv2.imshow('whatever', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create SURF object. You can specify params here or later.
surf = cv2.xfeatures2d.SURF_create(1609)

# Find keypoints and descriptors directly
# kp is array of all possible keypoints
# kp, des = surf.detectAndCompute(img, None)
keypointsList = surf.detect(img, None)

# surf.setUpright(True)

img2 = cv2.drawKeypoints(img, keypointsList, None, (255, 0, 0), 4)

potentialPointsList = rectArrayReturn(keypointsList)
# print(potentialPointsList[0].x, potentialPointsList[0].y, potentialPointsList[0].length, potentialPointsList[0].width)

for rect in potentialPointsList:
    print(rect.x, rect.y, rect.length, rect.width)
    img = cv2.rectangle(img, (int(rect.x), int(rect.y)), (int(rect.x + rect.length), int(rect.y + rect.width)), (255, 0, 0))
    
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 600,600)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.destroyAllWindows()
plt.imshow(img), plt.show()
