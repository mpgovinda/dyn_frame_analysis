import os
import sys
import numpy as np
import urllib.request as rq
import cv2

# scriptpath = "/Users/govinda/Workspace/project_snake/src/blob_detection.py"
#
# # Add the directory containing your module to the Python path (wants absolute paths)
# sys.path.append(os.path.abspath(scriptpath))
#
# import blob_detection


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = rq.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


# initialize the list of image URLs to download
urls = [
    "http://www.abc.net.au/news/image/6906366-3x2-940x627.jpg",

]


# loop over the image URLs
def download():
    for url in urls:
        # download the image URL and display it
        print ("downloading %s" % (url))
        image = url_to_image(url)
        cv2.imshow("Image", image)
        cv2.waitKey(0)




