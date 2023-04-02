import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from mtcnn import MTCNN
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to input video file")
args = vars(ap.parse_args())

# open the video file
cap = cv2.VideoCapture(args["video"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# initialize the MTCNN face detection model
detector = MTCNN()

# initialize the video stream
vs = VideoStream(src=0).start()

# define the initial blur size and the maximum blur size
init_blur_size = 5
max_blur_size = 50

while True:
    # read the frame from the video stream
    frame = vs.read()

    # detect faces in the frame using the MTCNN model
    results = detector.detect_faces(frame)

    if results:
        # get the bounding box of the first face
        x, y, w, h = results[0]['box']

        # calculate the center of the face
        cx, cy = x + w // 2, y + h // 2

        # calculate the distance from the center of the frame
        dx, dy = cx - frame.shape[1] // 2, cy - frame.shape[0] // 2

        # calculate the distance from the camera
        dist = w * 400 / frame.shape[1]

        # calculate the blur size based on the direction and speed of the person
        blur_size = int(init_blur_size + sigmoid((dx**2 + dy**2) /
                        (dist**2 + 1e-8)) * (max_blur_size - init_blur_size))

        # apply the blur to the face region
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (blur_size, blur_size), 0)
        frame[y:y+h, x:x+w] = blurred_face

    # show the resulting frame
    cv2.imshow('frame', frame)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# stop the video stream and close all windows
vs.stop()
cv2.destroyAllWindows()
