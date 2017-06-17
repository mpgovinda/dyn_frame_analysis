import cv2
import numpy as np
from matplotlib import pyplot as plt
import six


def img_pre_analysis():
    img = cv2.imread('stream/watch.jpg', cv2.IMREAD_COLOR)
    img[100:150, 100:150] = [0, 0, 0]
    px = img[55, 55]
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_adv_analysis():
    img1 = cv2.imread('stream/3d_demo_1.png')
    img2 = cv2.imread('stream/3d_demo_2.png')

    # add = img1 + img2
    # add = cv2.add(img1, img2)
    # add = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
    rows, cols, channels = img2.shape    # retrieve the rows, columns and the channel of the image
    roi = img1[0:rows, 0:cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('dst', dst)

    # cv2.imshow('add', add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def noiseReduction():
    img = cv2.imread('stream/horse.png')

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    plt.show()

# img_adv_analysis()
noiseReduction()