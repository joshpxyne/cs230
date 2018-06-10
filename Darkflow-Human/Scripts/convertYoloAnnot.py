"""
Modified from YOLO Annotation conversion script by Guanghan Ning, Josh Payne
Github: https://github.com/Guanghan/darknet/blob/master/scripts/convert.py

Modified by Peggy Wang
"""

import os
from os import walk, getcwd
from PIL import Image
from lxml import etree
import xml.etree.cElementTree as ET
import glob
import cv2
import numpy as np

classes = ["person"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def write_xml(folder, imgname, obj, x_min,x_max,y_min,y_max, width, height, number, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # height = 300
    # width = 300
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
    ET.SubElement(bbox, 'xmin').text = x_min
    ET.SubElement(bbox, 'ymin').text = y_min
    ET.SubElement(bbox, 'xmax').text = x_max
    ET.SubElement(bbox, 'ymax').text = y_max

    xml_str = ET.tostring(annotation)
    #print xml_str
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    #print "savedir", savedir
    imgname = imgname + str(number) + ".jpg"
    save_path = os.path.join(savedir, imgname.replace(".jpg", ".xml"))
    #print "save", save_path
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

"""-------------------------------------------------------------------"""

""" Configure Paths"""
mypath = "./aerialTrainAnnotBackup/"
outpath = "./Convert/"

cls = "person"
if cls not in classes:
    exit(0)
cls_id = classes.index(cls)

wd = getcwd()
#list_file = open('%s/%s_list.txt'%(wd, cls), 'w+')

""" Get input text file list """
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break
print(txt_name_list)

""" Process """
for txt_name in txt_name_list:
    # txt_file =  open("Labels/stop_sign/001.txt", "r")

    """ Open input text files """
    txt_path = mypath + txt_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"

    # """ Open output text files """
    # txt_outpath = outpath + txt_name
    # print("Output:" + txt_outpath)
    # txt_outfile = open(txt_outpath, "w+")

    ct = 0
    #convert data into PascalVOC XML format
    for i, line in enumerate(lines):
        #print i, line
        if i != 0 and len(line) > 1:
            ct = ct + 1
            elems = line.split(' ')
            #print(elems)
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            #print os.path.splitext(txt_name)[0]
            #print wd + "/aerialTest/" + os.path.splitext(txt_name)[0] + ".jpg"
            image = cv2.imread(wd + "/aerialTrain/" + os.path.splitext(txt_name)[0] + ".jpg")
            height = np.size(image, 0)
            width = np.size(image, 1)
            #print width, height
            write_xml("Images", os.path.splitext(txt_name)[0], "person", xmin,xmax,ymin,ymax, width, height, i, "Convert")
    #
    # """ Convert the data to YOLO format """
    # ct = 0
    # for line in lines:
    #     # print('lenth of line is: ')
    #     # print(len(line))
    #     # #print('\n')
    #     if(len(line) >= 5):
    #         ct = ct + 1
    #         print(line + "\n")
    #         elems = line.split(' ')
    #         print(elems)
    #         xmin = elems[0]
    #         xmax = elems[2]
    #         ymin = elems[1]
    #         ymax = elems[3]
    #         #
    #         print wd
    #         print os.path.splitext(txt_name)[0]
    #         img_path = str('%s/aerialTrain/%s.jpg'%(wd, os.path.splitext(txt_name)[0]))
    #         #t = magic.from_file(img_path)
    #         #wh= re.search('(\d+) x (\d+)', t).groups()
    #         im=Image.open(img_path)
    #         w= int(im.size[0])
    #         h= int(im.size[1])
    #         #w = int(xmax) - int(xmin)
    #         #h = int(ymax) - int(ymin)
    #         # print(xmin)
    #         print (w, h)
    #         print xmin
    #         b = (float(xmin), float(xmax), float(ymin), float(ymax))
    #         bb = convert((w,h), b)
    #         print(bb)
    #         txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    #
    # """ Save those images with bb into list"""
    # if(ct != 0):
    #     list_file.write('%s/aerialTrain/%s.jpg\n'%(wd, os.path.splitext(txt_name)[0]))

#list_file.close()
