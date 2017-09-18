import cv2
import numpy as np

img = cv2.imread('peda/10-8-09Route322PEDSAFETY031.jpg', 0)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()