import numpy as np
import cv2
import random
import math

def drawRectangle(img):
    # Off-White, Black, Grey, Red
    colors = [(233, 233, 233), (133, 133, 133), (35, 35, 200),(255,255,255)]
    point1 = (20,20)
    point2 = (280,280)
    random_color =  colors[3]
    # NOTE: Draws a rectangle
    cv2.rectangle(img, point1, point2, random_color, -1);

def drawSemiCircle(img):
    radius=130
    axes = (radius,radius)
    angle=0;
    startAngle=0;
    endAngle=180;
    center=(150,90)
    color=255
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, -1)

def drawQuarterCircle(img):
    radius=130
    axes = (radius,radius)
    angle=0;
    startAngle=0;
    endAngle=90;
    center=(90,90)
    color=255
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, -1)

def drawTriangle(img):
    triangle_center = [150, 150];
    height = 150;
    point_1 = [triangle_center[0], triangle_center[1] - height/2]
    point_2 = [triangle_center[0] + height/2, triangle_center[1] + height/2]
    point_3 = [triangle_center[0] - height/2, triangle_center[1] + height/2]
    pts = np.array([point_1, point_2, point_3], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], (255, 255, 255))

def drawTrapezoid(img):
    trapezoid_center = [150, 150];
    b1 = 150;
    b2 = 250;
    height = 100;
    point_1 = [trapezoid_center[0] - b1/2, trapezoid_center[1] - height/2]
    point_2 = [trapezoid_center[0] + b1/2, trapezoid_center[1] - height/2]
    point_3 = [trapezoid_center[0] - b2/2, trapezoid_center[1] + height/2]
    point_4 = [trapezoid_center[0] + b2/2, trapezoid_center[1] + height/2]
    pts = np.array([point_1, point_2, point_4, point_3], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)

def drawPentagon(img):
    pentagon_center = [150, 150];
    axis = 100;
    points = [];
    for i in range(0, 5):
        points.append([pentagon_center[0] + axis * math.cos(72 * i * 3.14/180), pentagon_center[1] + axis * math.sin(72 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)


def drawHexagon(img):
    pentagon_center = [150, 150];
    axis = 100;
    points = [];
    for i in range(0, 6):
        points.append([pentagon_center[0] + axis * math.cos(60 * i * 3.14/180), pentagon_center[1] + axis * math.sin(60 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)


def drawHeptagon(img):
    pentagon_center = [150, 150];
    axis = 100;
    points = [];
    for i in range(0, 7):
        points.append([pentagon_center[0] + axis * math.cos(51.4 * i * 3.14/180), pentagon_center[1] + axis * math.sin(51.4 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5], points[6]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)

def drawOctagon(img):
    pentagon_center = [150, 150];
    axis = 100;
    points = [];
    for i in range(0, 8):
        points.append([pentagon_center[0] + axis * math.cos(45 * i * 3.14/180), pentagon_center[1] + axis * math.sin(45 * i * 3.14/180)]);
    pts = np.array([points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)

def drawStar(img):
    pentagon_center = [150, 150];
    axis = 70;
    points = [];
    for i in range(0, 5):
        points.append([pentagon_center[0] + axis * math.cos((90 + 72 * -i) * 3.14/180), pentagon_center[1] + axis * math.sin((90 + 72 * -i) * 3.14/180)]);
    points2 = [];
    axis = 150;
    for i in range(0, 5):
        points2.append([pentagon_center[0] + axis * math.cos((-90 + 72 * -i) * 3.14/180), pentagon_center[1] + axis * math.sin((-90 + 72 * -i) * 3.14/180)]);
    pts = np.array([points[0], points2[3], points[1], points2[4], points[2], points2[0], points[3], points2[1], points[4], points2[2]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img,[pts], 255)

def drawCircle(img):
    # Off-White, Black, Grey, Red
    colors = [(233, 233, 233), (133, 133, 133), (35, 35, 200),(255,255,255)]
    random_color = colors[3]
    cv2.circle(img,(150,150),100,random_color,-1)


def padCharImg(data):
    padded_data = np.zeros((300,300,3),np.uint8)
    padded_data[100:200,100:200,0:3] = data
    return padded_data

def pixelDistFromBlack(pixel1):
    BLACK_RGB = [0,0,0]
    r_diff = (pixel1[0]-BLACK_RGB[0])*(pixel1[0]-BLACK_RGB[0])
    g_diff = (pixel1[1]-BLACK_RGB[1])*(pixel1[1]-BLACK_RGB[1])
    b_diff = (pixel1[2]-BLACK_RGB[2])*(pixel1[2]-BLACK_RGB[2])
    return math.sqrt(r_diff+g_diff+b_diff)

# Function is passed a blank white shape img + a black-on-white alphanum img
# NOTE: this function can pass an img background with black color (which wouldn't contrast with the already black background
#       surrounding the shape)
def colorShapeIMG(img,new_dst):
    WHITE_RGB = [255,255,255]
    # White, Black, Gray, Red, Blue, Green, Yellow, Purple, Brown, Orange
    colors = [(255,255,255),(0,0,0),(128,128,128),(255,0,0),(0,0,255),(0,128,0),(255,255,0),(128,0,128),(131,92,59),(255,165,0)]
    random_background_color = colors[int(random.random()*10)]
    random_alphanum_color = colors[int(random.random()*10)]
    # Makes sure the alpanum color and the background color are different
    while(random_background_color == random_alphanum_color):
        random_alphanum_color = colors[int(random.random()*10)]

    for i in range(0,len(img)):
        for x in range(0,len(img[i])):
            if np.all(img[i][x]==WHITE_RGB):
                if pixelDistFromBlack(new_dst[i][x]) < 200:
                    img[i][x] = random_alphanum_color
                else:
                    img[i][x] = random_background_color
    return img


def overlayChar(data, img):
    WHITE_RGB = [255,255,255]
    # Inverts the color
    data = 255-data;
    num = int(data.shape[0]*random.random())
    # Rotates the image between -10 to 10 degrees
    M = cv2.getRotationMatrix2D((32/2,32/2),180*(random.random()-.5),1)
    # NOTE: OpenCV transformation (warpAffine) Applies the transformation
    dst = cv2.warpAffine(data[num], M, (32,32))

    # NOTE: OpenCV resize transformation (handwritten character is slighly smaller than the background img)
    dst = cv2.resize(dst,(100,100),interpolation=cv2.INTER_NEAREST)

    new_dst = (dst, dst, dst)
    new_dst = np.stack(new_dst, axis=2);
    # places the smaller dst character image in a larger dst that matches the background img size, height+width
    new_dst = padCharImg(new_dst)

    # PREVIOUSLY: This is subtracted in order to convert the new_dst white (255,255,255) rgb values to black values (0,0,0)
    # return img-new_dst

    new_dst = 255-new_dst
    # Returns a random background / alphanumeric combination
    img = colorShapeIMG(img,new_dst)
    return img,new_dst



# Creates the alphanumeric "text" image array
data = np.load("alphanum-hasy-data-X.npy")

# Creates a black image array (height,width,rgb)
img = np.zeros((300, 300, 3), np.uint8);
drawTriangle(img);

# Overlays the "text" image over the "visual" image + modifies the image in orientation etc.
dst,secDST = overlayChar(data, img);
cv2.namedWindow("Test",cv2.WINDOW_NORMAL)
cv2.imshow("Test", dst);
cv2.waitKey(0);

cv2.namedWindow("Test2",cv2.WINDOW_NORMAL)
cv2.imshow("Test2",secDST)
cv2.waitKey(0)
