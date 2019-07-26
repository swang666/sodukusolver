import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

filename = 'puzzle1.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

outerBox = np.zeros(img.shape, np.uint8)

gray = cv2.GaussianBlur(gray, (5,5), 0)

outerBox = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

outerBox = cv2.bitwise_not(outerBox)

kernel = np.ones((3,3),np.uint8)
outerBox = cv2.dilate(outerBox, kernel)

ret,thresh = cv2.threshold(outerBox,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#out = cv2.drawContours(img, contours, -1, (100, 0 ,100), 3)

maxContour = max(contours, key=cv2.contourArea)

extLeft = tuple(maxContour[maxContour[:, :, 0].argmin()][0])
extRight = tuple(maxContour[maxContour[:, :, 0].argmax()][0])
extTop = tuple(maxContour[maxContour[:, :, 1].argmin()][0])
extBot = tuple(maxContour[maxContour[:, :, 1].argmax()][0])

# determine corner positions by quadrants
midpoint = ((extLeft[0] + extRight[0])/2, (extTop[1]+extBot[1])/2)
corners = [extLeft, extRight, extTop, extBot]
cornerLT = (0, 0)
cornerLB = (0, 0)
cornerRT = (0, 0)
cornerRB = (0, 0)
for point in corners:
    if point[0] < midpoint[0] and point[1] < midpoint[1]:
        cornerLT = point
    if point[0] < midpoint[0] and point[1] > midpoint[1]:
        cornerLB = point
    if point[0] > midpoint[0] and point[1] < midpoint[1]:
        cornerRT = point
    if point[0] > midpoint[0] and point[1] > midpoint[1]:
        cornerRB = point

# crop out puzzle and transform
pts1 = np.float32([cornerLT,cornerRT,cornerLB,cornerRB])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

cv2.imshow("Image", dst)
cv2.waitKey(0)