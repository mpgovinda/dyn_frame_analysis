import numpy as np
import cv2

img = cv2.imread('images/snakebite-girls-camp.jpg', 0)
img = cv2.medianBlur(img,1)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

# Increase intensive level
ret, img = cv2.threshold(cimg, 100, 255, cv2.THRESH_BINARY)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Detected Circle', im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

