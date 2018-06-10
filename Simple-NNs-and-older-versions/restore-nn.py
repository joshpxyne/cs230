import numpy as np
import pandas as pd
import random
import cv2

import keras as K

emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

num_classes = 47

tst = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
tst = -(255-tst)/255.0
nm = np.array(tst)
print(nm)

test_db  = pd.read_csv("data/emnist-balanced-test.csv")

y_test = test_db.iloc[:,0]
y_test = K.utils.np_utils.to_categorical(y_test, num_classes)

x_test = test_db.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255

# load json and create model
json_file = open('nn-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = K.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("nn-model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(loaded_model.evaluate(x_test, y_test, verbose=1))
nm = np.reshape(nm, (1,784))
prediction = loaded_model.predict(nm)
print(prediction)

pred = 0
for i in range(47):
	if prediction[0][i] > pred:
		pred = prediction[0][i]
		print(prediction[0][i])
		predIndex = i

print("Model's prediction of this character: " + emnist[predIndex]+", with a probability of " + str(100*pred)+"%")