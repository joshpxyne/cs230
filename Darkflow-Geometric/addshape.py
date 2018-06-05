import cv2
from scipy import ndimage
import numpy as np
from random import randint
import imutils
import csv
import random
import os
from PIL import Image
from lxml import etree
import xml.etree.cElementTree as ET


def write_xml(folder, imgname, obj, x_min,x_max,y_min,y_max, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    height = 300
    width = 300
    depth = 3

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = imgname + ".jpg"
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = obj
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(x_min)
    ET.SubElement(bbox, 'ymin').text = str(y_min)
    ET.SubElement(bbox, 'xmax').text = str(x_max)
    ET.SubElement(bbox, 'ymax').text = str(y_max)

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, imgname+".xml")
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


# for reference
# BLUE = [255,0,0]
# RED = [0, 0, 255]
# GREEN = [0, 255, 0]
# YELLOW = [0, 255, 255]

imgDim = 300

shapeNames = ["triangle","circle","parallelogram","plus","qcircle","rectangle","hcircle","square","star","rhombus"]
emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

shapes = []
for j in range(10):
	img = Image.open("shape-template/"+str(j)+".jpg")
	img = img.convert("RGBA")
	datas = img.getdata()

	newData = []
	for item in datas:
		if item[0] <= 200 and item[1] <= 200 and item[2] <= 200:
			newData.append((255, 255, 255, 0))
		else:
			newData.append(item)

	img.putdata(newData)
	shapes.append(img)
for s in range(10):
	shape = shapes[s]
	for m in range(1000): # todo: resize?, skew?
		ind=0
		charInd = randint(0,112000)
		char = []
		with open('emnist-balanced-train.csv', 'rb') as csvfile:
			charReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for i, row in enumerate(charReader):
				if i == charInd:
					char = row
					break

		char = char[0].split(",")
		alphanum = int(char[0])

		### Choose random colors ###

		randColor1 = randint(0,100)
		randColor2 = randint(0,100)
		randColor3 = randint(0,100)

		randColorIndex = randint(0,6)

		if randColorIndex == 0:
			COLOR = [255-randColor1,randColor2,randColor3]
			c = "b"

		if randColorIndex == 1:
			COLOR = [randColor2,randColor2,255-randColor1]
			c = "r"

		if randColorIndex == 2:
			COLOR = [randColor2,255-randColor1,randColor2]
			c = "g"

		if randColorIndex == 3:
			COLOR = [randColor2,255-randColor1,255-randColor1]
			c = "y"

		if randColorIndex == 4:
			COLOR = [127-randColor1,randColor2,randColor3]
			c = "db"

		if randColorIndex == 5:
			COLOR = [randColor2,randColor2,127-randColor1]
			c = "dr"

		if randColorIndex == 6:
			COLOR = [randColor2,127-randColor1,127-randColor1]
			c = "dy"

		nums = list()
		for index in range(0,6):
			if index != randColorIndex:
				nums.append(index)

		randColor2Index = random.choice(nums)

		if randColor2Index == 0:
			COLOR2 = [255-randColor1,randColor2,randColor3]
			c2 = "b"

		if randColor2Index == 1:
			COLOR2 = [randColor2,randColor2,255-randColor1]
			c2 = "r"

		if randColor2Index == 2:
			COLOR2 = [randColor2,255-randColor1,randColor2]
			c2 = "g"

		if randColor2Index == 3:
			COLOR2 = [randColor2,255-randColor1,255-randColor1]
			c2 = "y"

		if randColor2Index == 4:
			COLOR2 = [127-randColor1,randColor2,randColor3]
			c2 = "db"

		if randColor2Index == 5:
			COLOR2 = [randColor2,randColor2,127-randColor1]
			c2 = "dr"

		if randColor2Index == 6:
			COLOR2 = [randColor2,127-randColor1,127-randColor1]
			c2 = "dy"

		img = shape
		ind = 1
		mod = randint(-2,2)
		letArr = np.array(char, dtype=np.uint32)
		letArr = np.delete(letArr, 0)
		letArr = np.reshape(letArr,(28,28))
		let = Image.fromarray(letArr)
		til = Image.new("RGBA",(100,100))
		let = let.convert("RGBA")
		til.paste(let,(35,35))
		datas = til.getdata()
		datas2 = img.getdata()
		newData = []
		j = 0
		for item in datas:

			if item[0] >= 50 and item[1] >= 50 and item[2] >= 50:
				newData.append((max(min(COLOR[0] + randint(-20,20),255),0), 
								max(min(COLOR[1] + randint(-20,20),255),0), 
								max(min(COLOR[2] + randint(-20,20),255),0)))
			else:
				if datas2[j][3]>=0.5:
					newData.append((max(min(COLOR2[0] + randint(-20,20),255),0), 
									max(min(COLOR2[1] + randint(-20,20),255),0), 
									max(min(COLOR2[2] + randint(-20,20),255),0)))
				else:
					newData.append(datas2[j])

			j+=1
		img.putdata(newData)
		randDim = random.randint(80,150)
		img = img.rotate(random.randint(0,359), expand=1).resize((randDim,randDim))		
		bg = Image.open("backgrounds/"+random.choice(os.listdir("backgrounds")))
		bg = bg.convert("RGBA")
		centerX = randint(-100,100)
		centerY = randint(-100,100)
		offset = (100+centerX, 100+centerY)
		x_min = max(0,100+centerX)
		y_min = max(0,100+centerY)
		x_max = min(x_min + randDim, 300)
		y_max = min(y_min + randDim, 300)
		bg.paste(img, offset, img)
		bg = bg.convert("RGB")
		bg.save("data2/"+shapeNames[s]+str(m)+".jpg")
		xml_string = write_xml("data2", shapeNames[s]+str(m), shapeNames[s],x_min,x_max,y_min,y_max, "xml")