import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')


def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 81), 2)
    return frame


def func():
    while True:
        ret, frame = cap.read()
        # Resizing down the image to fit in the screen.
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # adding a border
        dst = cv2.copyMakeBorder(dst, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        # calling face detection function
        dst = face_detection(dst)

        # creating another frame.
        channels = cv2.split(dst)
        frame_merge = cv2.merge(channels)

        # horizintally concatenating the two frames.
        final_frame = cv2.hconcat((frame, frame_merge))

        # Show the concatenated frame using imshow.
        cv2.imshow('frame', final_frame)

        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
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
    plt.subplot(131), plt.imshow(gray[2], 'gray')
    plt.subplot(132), plt.imshow(noisy[2], 'gray')
    plt.subplot(133), plt.imshow(dst, 'gray')
    plt.show()

func()
# filter()