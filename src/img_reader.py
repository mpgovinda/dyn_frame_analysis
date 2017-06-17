import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('stream/watch.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread('stream/watch.jpg', cv2.IMREAD_COLOR)
cv2.line(img, (0, 0), (150, 150), (0, 255, 0), 5)                   # plotting line, rectangle, circle
cv2.rectangle(img, (60, 60), (400, 400), (0, 255, 255), 5)
cv2.circle(img, (250, 250), 100, (255, 0, 0), 5)

pts = np.array([[0, 10], [20, 30], [50, 30], [100, 120]], np.int32)
cv2.polylines(img, [pts], True, (0, 0, 255), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Welcome to OpenCv', (0, 200), font, 1, (0, 0, 0), 4, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

