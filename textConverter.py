import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
from cvzone.ClassificationModule import Classifier

# cap = cv2.imread("yokesh.jpg")
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
labels = ["A", "B", "C", "D", 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


offset = 20
imgSize = 300

folder = "data/A"
counter = 0

while True:
    img = cv2.imread("yokesh.jpg")
    imgOutput = img.copy()
    hands,_ = detector.findHands(img)

    if hands:
        try:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y-20: y+h + 20, x-20: x+w+ 20]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            
            if aspectRatio > 1:
                k = imgSize/h 
                wcal = math.ceil(k*w)

                imgResize = cv2.resize(imgCrop, (wcal, imgSize))
                imgResizeShape = imgResize.shape

                wGap = math.ceil((imgSize - wcal)/2)

                imgWhite[:, wGap:wcal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            
            else:
                k = imgSize/w 
                hCal = math.ceil(k*h)

                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape

                hGap = math.ceil((imgSize - hCal)/2)

                imgWhite[hGap:hCal+hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)

            cv2.rectangle(imgOutput, (x-offset, y-offset -50), (x-offset + 50, y-offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x - offset + 8, y - offset - 5), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
            
            cv2.rectangle(imgOutput, (x - offset,y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("imgWhite", imgWhite)
            cv2.imshow("imgCrop", imgCrop)
        except Exception as e:
            continue



    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord("s"):

        counter += 1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)