import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

filename = 'puzzle1.png'
img = cv2.imread(filename)
""" gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None) """

# Threshold for an optimal value, it may vary depending on the image.
img[32,404]=[0,0,255]
img[404,417]=[0,0,255]
img[43,32]=[0,0,255]
img[417,43]=[0,0,255]

#A = np.where(dst>0.3*dst.max())
#B = np.asarray(A).T
#minx, miny, maxx, maxy = B[0][0], B[0][1],B[0][0], B[0][1]
""" for pair in B:
    if minx > pair[0]:
        minx = pair[0]
    if miny > pair[1]:
        miny = pair[1]
    if maxx < pair[0]:
        maxx = pair[0]
    if maxy < pair[1]:
        maxy = pair[1]

upperleft = [x for x in B if x[0] == minx]
bottomleft = [x for x in B if x[1] == miny]
upperright = [x for x in B if x[1] == maxy]
bottomright = [x for x in B if x[0] == maxx]
print(upperleft,upperright,bottomleft,bottomright) """

rows,cols,ch = img.shape

pts1 = np.float32([[32,404],[404,417],[43,32]])
pts2 = np.float32([[32,404],[404,404],[32,32]])

M = cv2.getRotationMatrix2D((cols/2,rows/2),-math.degrees(math.atan(13/372)),1)
dst = cv2.warpAffine(img,M,(cols,rows))
""" M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))
 """
""" M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))
 """
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

""" cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows() """