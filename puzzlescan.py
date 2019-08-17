import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import sys

def puzzle_process(filename):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h = 10)

        outerBox = np.zeros(img.shape, np.uint8)

        gray = cv2.GaussianBlur(gray, (5,5), 0)


        outerBox = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

        outerBox = cv2.bitwise_not(outerBox)

        kernel = np.ones((3,3),np.uint8)
        outerBox = cv2.dilate(outerBox, kernel)

        ret,thresh = cv2.threshold(outerBox,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        maxContour = max(contours, key=cv2.contourArea)

        extLeft = tuple(maxContour[maxContour[:, :, 0].argmin()][0])
        extRight = tuple(maxContour[maxContour[:, :, 0].argmax()][0])
        extTop = tuple(maxContour[maxContour[:, :, 1].argmin()][0])
        extBot = tuple(maxContour[maxContour[:, :, 1].argmax()][0])
        midpoint = (int((extLeft[0] + extRight[0])/2), int((extTop[1]+extBot[1])/2))

        first_quadrant = [p[0] for p in maxContour if (p[0][0] > midpoint[0]) and (p[0][1] < midpoint[1])]
        second_quadrant = [p[0] for p in maxContour if (p[0][0] < midpoint[0]) and (p[0][1] < midpoint[1])]
        third_quadrant = [p[0] for p in maxContour if (p[0][0] < midpoint[0]) and (p[0][1] > midpoint[1])]
        fourth_quadrant = [p[0] for p in maxContour if (p[0][0] > midpoint[0]) and (p[0][1] > midpoint[1])]

        def distance(p):
                return (p[0] - midpoint[0]) ** 2 + (p[1] - midpoint[1]) ** 2 

        # determine corner positions by quadrants
        cornerRT = tuple(max(first_quadrant, key = distance))
        cornerLT = tuple(max(second_quadrant, key = distance))
        cornerLB = tuple(max(third_quadrant, key = distance))
        cornerRB = tuple(max(fourth_quadrant, key = distance))


        # crop out puzzle and transform
        pts1 = np.float32([cornerLT,cornerRT,cornerLB,cornerRB])
        pts2 = np.float32([[0,0],[360,0],[0,360],[360,360]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        gray_dst = cv2.warpPerspective(outerBox,M,(360,360))

        # dilute grid lines
        '''
        gray_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray_dst = cv2.fastNlMeansDenoising(gray_dst, h = 10)
        gray_dst = cv2.GaussianBlur(gray_dst, (5,5), 0)
        gray_dst = cv2.adaptiveThreshold(gray_dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 5, 2)
        gray_dst = cv2.bitwise_not(gray_dst)
        '''

        horizontal = np.copy(gray_dst)
        vertical = np.copy(gray_dst)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = int(cols / 10)
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        # [horiz]
        # [vert]
        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = int(rows / 10)

        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        # Show extracted vertical lines

        ret_v,thresh_v = cv2.threshold(vertical,127,255,0)
        contours_v, hierarchy_v = cv2.findContours(thresh_v,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ret_h,thresh_h = cv2.threshold(horizontal,127,255,0)
        contours_h, hierarchy_h = cv2.findContours(thresh_h,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gray_dst, contours_v, -1, (0,0,0), 3)
        cv2.drawContours(gray_dst, contours_h, -1, (0,0,0), 3)

        # [vert]
        # [smooth]
        smooth = cv2.blur(gray_dst, (2, 2))
        smooth = cv2.fastNlMeansDenoising(smooth, h = 10)
        return smooth

def main():
    '''
    Train the model defined above.
    '''
    cv2.imshow("image", puzzle_process('puzzles/puzzle1.jpg'))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()


