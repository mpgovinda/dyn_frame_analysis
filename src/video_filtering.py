import cv2
import urllib
import numpy as np


def filtering():
    cap = cv2.VideoCapture('stream/ABUS.mp4')
    while True:
        grabbed, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([140, 140, 0])
        upper_red = np.array([255, 255, 255])

        rng = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=rng)

        kernel = np.ones((5, 5), np.uint8)
        # erosion = cv2.erode(res, kernel, iterations=1)
        # dilation = cv2.dilate(res,kernel, iterations=1)
        # smoothed = cv2.filter2D(res, -1, kernel)
        opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        median = cv2.medianBlur(res, 15)

        # cv2.imshow('smoothed', smoothed)
        cv2.imshow('frame', frame)
        cv2.imshow('res', res)
        # cv2.imshow('erosion', erosion)
        # cv2.imshow('dilation', dilation)
        cv2.imshow('median', median)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        if not grabbed or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


filtering()


