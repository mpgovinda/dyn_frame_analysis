import cv2
# from cv2 import cv
import os
import numpy as np

def showme(pic):
    cv2.imshow('window', pic)
    cv2.waitKey()
    cv2.destroyAllWindows()


# im=cv2.imread('images/1.jpg')
im=cv2.imread('images/snakebite-girls-camp.jpg')
# im=cv2.imread('images/images.jpg')


gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#I've tried blur,bw,tr...  all give me poor results.

blur = cv2.GaussianBlur(gray,(3,3),0)
n,bw = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
tr=cv2.adaptiveThreshold(blur,255,0,1,11,2)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 100, None, 200, 50, 2, 10)

try:
    n = np.shape(circles)
    circles = np.reshape(circles, (n[1], n[2]))
    print(circles)
    for circle in circles:
        cv2.circle(im, (circle[0], circle[1]), circle[2], (0, 255, 0))
    showme(im)
except:
    print("no cicles found")