import cv2
from scipy import ndimage
for i in range(10):
    print(i)
    img = cv2.imread("field-images/"+str(i)+".jpg")
    j = 0
    height, width, channels = img.shape 
    xScale = int((width)/50)
    yScale = int((height)/50)
    for x in range(xScale):
        for y in range(yScale):
            crop_img = img[50*y:50*y+300, 50*x:50*x+300]
            h,w,channels=crop_img.shape
            if h==300 and w==300:
                cv2.imwrite("backgrounds/background"+str(i)+str(j)+".png", crop_img)
                j += 1

# Next step: 
#   for image in ./backgrounds:
#       s = np.random.randn(10)
#       shape = cv.imread ("shape"+str(s))
#       newShape = keras.rotate(shape, random degrees)
#       r = np.random.randn(500000) # about 700000 images in the EMNIST, less if we only count capital letters and numbers
#       name = EMNIST[r].name
#       letter = EMNIST[r].image
#       shape = cv2.copyTo(keras.rotate(letter, random degrees), shape, NO_TRANSLATION,NO_ROTATION)
#       newImage = cv2.copyTo(shape,image,SOME_TRANSLATION,SOME_ROTATION)
#       cv2.imwrite("../newimgs/"+"letter"+letter+"shape"+str(s)+"index"+str(index)+".png",newImage)

