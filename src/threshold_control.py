import cv2
import numpy as np


def threshold_analysis():
    img = cv2.imread('stream/bookpage.jpg')
    # img = cv2.VideoCapture(0)
    retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval2, threshold2 = cv2.threshold(gray_img, 12, 255, cv2.THRESH_BINARY)
    gaus = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    cv2.imshow('img', img)
    cv2.imshow('threshold', threshold)
    cv2.imshow('threshold2', threshold2)
    cv2.imshow('gaus', gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

threshold_analysis()