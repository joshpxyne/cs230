import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-voc-10c.cfg',
    'load': 1250,
    'threshold': 0.05,
}


tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]


def runit(name):
	frame = cv2.imread(name)
	results = tfnet.return_predict(frame)
	maxconfidence = 0
	label = ""
	for color, result in zip(colors, results): # comment out for multi
		if result['confidence'] > maxconfidence:
			maxconfidence = result['confidence']
			color = color
			result = result
			tl = (result['topleft']['x'], result['topleft']['y'])
			br = (result['bottomright']['x'], result['bottomright']['y'])
			label = result['label']
			confidence = result['confidence']
			text = '{}: {:.0f}%'.format(label, confidence * 100)
	found = 0
	return [tl,br]
	# frame = cv2.rectangle(frame, tl, br, color, 5)
	# frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
	# cv2.imshow('frame', frame)
	# cv2.waitKey(0)
hcircle = qcircle = circle = plus = rhombus = parallelogram = triangle = rectangle = star = square = shape = 0
perc = 1000



for i in range(perc):
	print(i)
	hcircle += runit("data2/hcircle"+str(i)+".jpg","hcircle")[0]
	qcircle += runit("data2/qcircle"+str(i)+".jpg","qcircle")[0]
	circle += runit("data2/circle"+str(i)+".jpg","circle")[0]
	plus += runit("data2/plus"+str(i)+".jpg","plus")[0]
	rhombus += runit("data2/rhombus"+str(i)+".jpg","rhombus")[0]
	parallelogram += runit("data2/parallelogram"+str(i)+".jpg","parallelogram")[0]
	triangle += runit("data2/triangle"+str(i)+".jpg","triangle")[0]
	rectangle += runit("data2/rectangle"+str(i)+".jpg","rectangle")[0]
	star += runit("data2/star"+str(i)+".jpg","star")[0]
	square += runit("data2/square"+str(i)+".jpg","square")[0]
	shape += runit("data2/hcircle"+str(i)+".jpg","hcircle")[1]
	shape += runit("data2/qcircle"+str(i)+".jpg","qcircle")[1]
	shape += runit("data2/circle"+str(i)+".jpg","circle")[1]
	shape += runit("data2/plus"+str(i)+".jpg","plus")[1]
	shape += runit("data2/rhombus"+str(i)+".jpg","rhombus")[1]
	shape += runit("data2/parallelogram"+str(i)+".jpg","parallelogram")[1]
	shape += runit("data2/triangle"+str(i)+".jpg","triangle")[1]
	shape += runit("data2/rectangle"+str(i)+".jpg","rectangle")[1]
	shape += runit("data2/star"+str(i)+".jpg","star")[1]
	shape += runit("data2/square"+str(i)+".jpg","square")[1]
print("---------")
print(shape)
print(hcircle)
print(qcircle)
print(circle)
print(plus)
print(rhombus)
print(parallelogram)
print(triangle)
print(rectangle)
print(star)
print(square)
print("---------")
print("overall shapes: "+str(shape/10000.0))
print("hcircle: "+str(hcircle/1000.0))
print("qcircle: "+str(qcircle/1000.0))
print("circle: "+str(circle/1000.0))
print("plus: "+str(plus/1000.0))
print("rhombus: "+str(rhombus/1000.0))
print("parallelogram: "+str(parallelogram/1000.0))
print("triangle: "+str(triangle/1000.0))
print("rectangle: "+str(rectangle/1000.0))
print("star: "+str(star/1000.0))
print("square: "+str(square/1000.0))
