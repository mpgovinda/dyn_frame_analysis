import cv2
import numpy as np

# Read image
# im = cv2.imread("images/1.jpg")
# im = cv2.imread('images/snakebite-girls-camp.jpg', 0)
im = cv2.imread('images/snake bites.jpg')

# Increase intensive level
ret, im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Removing noise
# im_with_keypoints = cv2.fastNlMeansDenoisingColored(im_with_keypoints_pre, None, 10, 10, 7, 21)

imgray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
im_gauss = cv2.GaussianBlur(imgray, (5, 5), 0)
ret, thresh = cv2.threshold(im_gauss, 127, 255, 0)
# get contours
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours_area = []

for con in contours:
    area = cv2.contourArea(con)
    if 1000 < area < 10000:
        contours_area.append(con)
cv2.drawContours(im_with_keypoints, contours_area, -1, (0,255,0), 3)

# contours_area = []
# # calculate area and filter into new array
# for con in contours:
#     area = cv2.contourArea(con)
#     if 1000 < area < 10000:
#         contours_area.append(con)

# # Show blobs
# cv2.imshow("Pre", im_with_keypoints_pre)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)