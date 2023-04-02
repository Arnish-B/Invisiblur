import cv2
import numpy as np
import math


def invisiblur(the_file_name):
    def sigmoid(x):
        return abs((1 / (1 + math.exp(-x))) - 0.5) / 10

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(the_file_name)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    img_array = []
    # Read until video is completed
    count = 0
    original = []
    flag = 1
    last_even_p1 = 0
    last_even_p2 = 0
    last_even_p3 = 0
    last_even_p4 = 0
    last_odd_p1 = 0
    last_odd_p2 = 0
    last_odd_p3 = 0
    last_odd_p4 = 0
    flag = 0
    flag2 = 0
    not_updated = 0
    update_last = ""
    f = 0
    while cap.isOpened():
        flag += 1
        f += 1
        # if f == 145:
        #     break
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            if count == 0:
                count = 1
                original = img
            # Display the resulting frame
            # cv2.imshow("Frame", img)

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord("q"):
            #     break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if (
                len(faces) == 0
                and last_even_p1 != 0
                and last_odd_p1 != 0
                and not_updated < 16 * 0.5
            ):
                not_updated += 1
                if not_updated < 16 * 0.5:
                    flag2 = 1
                n = 1
                t = 25
                if abs(last_even_p1 - last_odd_p1) > t:
                    n = 1 + sigmoid(abs(last_even_p1 - last_odd_p1))
                d1 = int((last_even_p1 - last_odd_p1) * not_updated**n)
                if abs(last_even_p2 - last_odd_p2) > t:
                    n = 1 + sigmoid(abs(last_even_p2 - last_odd_p2))
                d2 = int((last_even_p2 - last_odd_p2) * not_updated**n)
                if abs(last_even_p3 - last_odd_p3) > t:
                    n = 1 + sigmoid(abs(last_even_p3 - last_odd_p3))
                d3 = int((last_even_p3 - last_odd_p3) * not_updated**n)
                if abs(last_even_p4 - last_odd_p4) > t:
                    n = 1 + sigmoid(abs(last_even_p4 - last_odd_p4))
                d4 = int((last_even_p4 - last_odd_p4) * not_updated**n)

                if update_last == "odd":
                    d1 *= -1
                    d2 *= -1
                    d3 *= -1
                    d4 *= -1
                img[p1 + d1-100: p2 + d2 + 20, p3 + d3 - 1000: p4 + d4 + 1400] = original[
                    p1 + d1-100: p2 + d2 + 20, p3 + d3 - 1000: p4 + d4 + 1400
                ]
                # img[p1+d1:p2+d2, p3+d3:p4+d4] = orignal[p1+d1:p2+d2, p3+d3:p4+d4]
            # if len(faces) != 0 and flag2 == 1:
            #     last_even_p1 = []
            #     last_even_p2 = []
            #     last_even_p3 = []
            #     last_even_p4 = []
            #     last_odd_p1 = []
            #     last_odd_p2 = []
            #     last_odd_p3 = []
            #     last_odd_p4 = []
            for (x, y, w, h) in faces:
                not_updated = 0

                p1 = y
                p2 = y + h
                p3 = x
                p4 = x + w
                img[p1-30:p2+20, p3-60:p4+40] = original[p1-30:p2+20, p3-60:p4+40]
                # p4 right edge
                # p3 left edge
                # p2 bottom edge
                # p1 upper edge
                if flag == 1:
                    # print(len(ROI), len(ROI[0]))
                    flag = 0
                if flag % 2 == 0:
                    update_last = "even"
                    last_even_p1 = p1
                    last_even_p2 = p3
                    last_even_p3 = p3
                    last_even_p4 = p4
                else:
                    update_last = "odd"
                    last_odd_p1 = p1
                    last_odd_p2 = p3
                    last_odd_p3 = p3
                    last_odd_p4 = p4
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    print(len(img_array))
    # Closes all the frames
    cv2.destroyAllWindows()
    out = cv2.VideoWriter(
        "video_test_processed.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
