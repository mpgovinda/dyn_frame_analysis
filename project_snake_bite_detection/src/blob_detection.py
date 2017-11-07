import cv2
import numpy as np
import math
import urllib
import io
import urllib.request
import argparse


array_x = []
array_y = []

# Test images
# img = cv2.imread('images/images.jpg', 0)
# img = cv2.imread('images/snake bites.jpg', 0)
# img = cv2.imread('images/snakebite-girls-camp.jpg', 0)


# initialize the list of image URLs to download
# url = "https://i.pinimg.com/736x/2a/83/a6/2a83a6fc2c83781de96c1cb2f83170fa--snakebite-girls-camp.jpg"
url = "http://www.abc.net.au/news/image/6906366-3x2-940x627.jpg"
# url = "http://www.telegraph.co.uk/content/dam/news/2016/10/23/snake-xlarge_trans_NvBQzQNjv4BqjJeHvIwLm2xPr27m7LF8meIXyYL3FtpSLfbzyKLr2lI.jpg"


def distance(array_x, array_y, cimg):
    try:
        y = int(abs(array_x[1] - array_x[0])**2) + int(abs(array_y[1] - array_y[0])**2)
        y = math.sqrt(y)

        # draw the line between points
        cv2.line(cimg, (array_x[0], array_y[0]), (array_x[1], array_y[1]), (0, 255, 0), 2)

        # print line length
        label = "{} px".format(str(y))
        cv2.putText(cimg, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 155, 81), 2)
    except IndexError:
        print('Bite does not found!')


def read_bite():
    # download the image URL and display it
    urllib.request.urlretrieve(url, "images/im.jpg")

    img = cv2.imread('images/im.jpg', 0)

    img = cv2.medianBlur(img, 5)

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ret, cimg = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=12, minRadius=0, maxRadius=10)

    # detect the circles
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        x, y = i[0], i[1]
        array_x.append(x)
        array_y.append(y)

    distance(array_x, array_y, cimg)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

read_bite()

