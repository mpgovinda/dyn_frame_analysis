import cv2
import numpy as np
from matplotlib import pyplot as plt


def func():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # Resizing down the image to fit in the screen.
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # creating another frame.
        channels = cv2.split(dst)
        frame_merge = cv2.merge(channels)

        # horizintally concatenating the two frames.
        final_frame = cv2.hconcat((frame, frame_merge))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame', final_frame)
        
        k = cv2.waitKey(30) & 0xff
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

func()