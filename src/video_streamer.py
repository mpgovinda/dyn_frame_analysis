import cv2
import numpy as np


def prime():
    cap = cv2.VideoCapture(0)  # capture the web cam video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # setup video format
    output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # setting up the output video
    while True:
        sta, frame = cap.read()  # starting to read(view) frame one by one
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # convert frame color to gray
        output.write(frame)  # save the video stream
        cv2.imshow('frame', frame)  # show the original videoq
        cv2.imshow('gray', gray)  # show the color converted video

        if not sta or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cap.destroyAllWindows()
