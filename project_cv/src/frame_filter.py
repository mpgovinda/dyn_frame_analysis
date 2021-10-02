# Code is tempred for new feature. New learning curve for a deep analysis of the images
# Opencv will upgrade in near future
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
# from matplotlib import pyplot as plt
import os

path = os.path.join(os.path.abspath, 'src')
print(path)

# Cascade files (Classifers)
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
side_face_cas = cv2.CascadeClassifier('xml/lbpcascade_profileface.xml')
upper_body_cascade = cv2.CascadeClassifier('xml/haarcascade_upperbody.xml')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Capture the video feed
cap = cv2.VideoCapture('peda/4p-c0.mp4')

# Video saving function
out = cv2.VideoWriter('output.avi', -1, 20.0, (300, 240))



def face_detection(frame):
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = side_face_cas.detectMultiScale(gray)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 81), 2)
        # save the rectangle face
        sub_face = frame[y:y+h, x:x+w]
        face_name = "faces/face_"+str(y)+".jpg"
        cv2.imwrite(face_name, sub_face)
    return frame


def body_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    body = upper_body_cascade.detectMultiScale(frame)

    for(x, y, w, h) in body:
        cv2.rectangle(body, (x, y), (x+w, y+h), (0, 200, 81), 2)
        # save the upper body
        # sub_body = frame[y:y+h, x:x+w]
        # upper_body_name = "upper_body/upper_"+str(y)+".jpg"
        # cv2.imwrite(upper_body_name, sub_body)
    return frame


def func():
    while True:
        ret, frame = cap.read()
        # Resizing down the image to fit in the screen.
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Denoise the frame
        dst = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

        # adding a border
        dst = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(dst, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # calling face detection function
        # dst = face_detection(dst)

        # calling upper body detection function
        # dst = body_detection(dst)

        # creating another frame.
        channels = cv2.split(dst)
        frame_merge = cv2.merge(channels)

        # horizintally concatenating the two frames.
        final_frame = cv2.hconcat((frame, frame_merge))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame', final_frame)
        # cv2.imshow('frame', frame)

        if not ret or cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def filter():
    cap = cv2.VideoCapture(0)
    # create a list of first 5 frames
    img = [cap.read()[1] for i in range(5)]

    # convert all to grayscale
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

    # convert all to float64
    gray = [np.float64(i) for i in gray]

    # create a noise of variance 25
    noise = np.random.randn(*gray[1].shape) * 10

    # Add this noise to images
    noisy = [i + noise for i in gray]

    # Convert back to uint8
    noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]

    # Denoise 3rd frame considering all the 5 frames
    dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
    # plt.subplot(131), plt.imshow(gray[2], 'gray')
    # plt.subplot(132), plt.imshow(noisy[2], 'gray')
    # plt.subplot(133), plt.imshow(dst, 'gray')
    # plt.show()

func()
# filter()