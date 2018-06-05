import numpy as np
import cv2
import os
import scipy.ndimage.morphology as mp
from PIL import Image

np.set_printoptions(threshold=np.nan)

for f in os.listdir("img2"):
	gray = cv2.imread("img2/"+f,0)
	h,w = gray.shape
	ret,binarized_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	n = -((np.array(binarized_img)/255)-1)
	img = Image.fromarray(np.uint8(n * 255) , 'L')
	mat = np.array(img)
	left = right = top = bottom = 0
	for col_num in range(w):
		if 255 in mat[:, col_num]:
			right = col_num
			if left == 0:
				left = col_num
	for row_num in range(h):
		if 255 in mat[row_num, :]:
			bottom = row_num
			if top == 0:
				top = row_num

	cropped = img.crop((left, top, left+max(right-left, bottom-top), top + max(right-left, bottom-top)))
	padded = np.array(cropped)/255
	shape = padded.shape
	padded = np.vstack([padded, np.zeros((5,shape[0]))])
	padded = np.vstack([np.zeros((5,shape[0])), padded])
	padded = np.hstack([padded, np.zeros((shape[0]+10,5))])
	padded = np.hstack([np.zeros((shape[0]+10,5)), padded])
	img = Image.fromarray(np.uint8(padded * 255) , 'L')
	img.save('shape/'+f)

for f in os.listdir("img3"):
	gray = cv2.imread("img3/"+f,0)
	h,w = gray.shape
	ret,binarized_img = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	n = -((np.array(binarized_img)/255)-1)
	img = Image.fromarray(np.uint8(n * 255) , 'L')
	mat = np.array(img)
	left = right = top = bottom = 0
	for col_num in range(w):
		if 255 in mat[:, col_num]:
			right = col_num
			if left == 0:
				left = col_num
	for row_num in range(h):
		if 255 in mat[row_num, :]:
			bottom = row_num
			if top == 0:
				top = row_num

	cropped = img.crop((left, top, left+max(right-left, bottom-top), top + max(right-left, bottom-top)))
	padded = np.array(cropped)/255
	shape = padded.shape
	padded = np.vstack([padded, np.zeros((5,shape[0]))])
	padded = np.vstack([np.zeros((5,shape[0])), padded])
	padded = np.hstack([padded, np.zeros((shape[0]+10,5))])
	padded = np.hstack([np.zeros((shape[0]+10,5)), padded])
	img = Image.fromarray(np.uint8(padded * 255) , 'L')
	img.save('letter/'+f)