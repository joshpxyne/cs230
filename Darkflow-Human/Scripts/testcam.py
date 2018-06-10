import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 2375, # replace with current checkpoint
    'threshold': 0.1,
    'gpu': 1.0
}

tfnet = TFNet(options)
imgcv = cv2.imread("./sample_img/henrique.jpg")
result = tfnet.return_predict(imgcv)
for thing in result:
    print(thing)
    label = thing['label']
    confidence = thing['confidence']
    topleft = thing['topleft']
    bottomright = thing['bottomright']
    cv2.rectangle(imgcv,(topleft['x'],topleft['y']),(bottomright['x'],bottomright['y']),(0,255,0),3)
    #cv2.putText(imgcv, label + "confidence: " + str(confidence), (topleft['x'],topleft['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2,cv2.LINE_AA)

cv2.imshow('pic', imgcv)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(result)

# colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
#
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#
# while True:
#     stime = time.time()
#     ret, frame = capture.read()
#     if ret:
#         results = tfnet.return_predict(frame)
#         for color, result in zip(colors, results):
#             tl = (result['topleft']['x'], result['topleft']['y'])
#             br = (result['bottomright']['x'], result['bottomright']['y'])
#             label = result['label']
#             confidence = result['confidence']
#             text = '{}: {:.0f}%'.format(label, confidence * 100)
#             frame = cv2.rectangle(frame, tl, br, color, 5)
#             frame = cv2.putText(
#                 frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
#         cv2.imshow('frame', frame)
#         print('FPS {:.1f}'.format(1 / (time.time() - stime)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# capture.release()
# cv2.destroyAllWindows()
