from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2


class VideoLoading:
    def parser(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", required=True, help="path to input file")
        args = vars(ap.parse_args())

    def streamer(self):
        stream = cv2.VideoCapture(self.args["video"])
        fps = FPS().start()

        while True:

            # grab the frame from the threaded video file stream
            (grabbed, frame) = stream.read()

            if not grabbed:
                break

            # resize the frame and convert into grayscale (retaining 3 channels)
            frame = imutils.resize(frame, width=450)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.dstack([frame, frame, frame])

            # display a piece of text to the frame( can benchmark fairly against the fast method)
            cv2.putText(frame, "slow method", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # show the frame and update the FPS counter
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            fps.update()


def main():
    vl = VideoLoading()
    vl.parser()
    vl.videoStream()


if __name__ == "main":
    main()
