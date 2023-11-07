import cv2
import numpy as np
import math


def blurThis(the_fileName):
    def sigmoid(x):
        return abs((1 / (1 + math.exp(-x))) - 0.5) / 10

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    # cap = cv2.VideoCapture("smol_video_test.mp4")
    cap = cv2.VideoCapture(the_fileName)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    img_array = []
    size = 0
    # Read until video is completed
    count = 0
    original = []
    threshold = 8
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
                height, width, layer = img.shape
                count = 1
                original = img.copy()
                nonBlured_original = original.copy()
                grid = [[1 for i in range(6)] for i in range(6)]
                a = 15
                kernel = np.ones((a, a), dtype=np.float32) / (a**2)

                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                convolved = cv2.filter2D(
                    gray, -1, kernel, borderType=cv2.BORDER_REPLICATE
                )
                convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
                # Crop the result to the same size as the input frame
                height, width = original.shape[:2]
                original = convolved[:height, :width]
                # print(convolved.shape, original.shape)
                # cv2.imshow("Blurred Image", convolved)
                # cv2.waitKey(1)

            # Display the resulting frame
            # cv2.imshow("Frame", img)

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord("q"):
            #     break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for x, y, w, h in faces:
                not_updated = 0
                t = 20

                p1 = max(y - t, 0)
                p2 = min(y + h + t, height)
                p3 = max(x - t, 0)
                p4 = min(x + w + t, width)

                kernel = np.ones((a, a), dtype=np.float32) / a**2

                subframe = img[p1:p2, p3:p4]
                gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
                convolved = cv2.filter2D(
                    gray, -1, kernel, borderType=cv2.BORDER_REPLICATE
                )
                convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
                original_sub_section = original[p1:p2, p3:p4]
                nonBlured_original_subSection = nonBlured_original[p1:p2, p3:p4]
                diff = np.abs(convolved - original_sub_section)
                mask = diff > threshold
                convolved[mask] = nonBlured_original_subSection[mask]
                img[p1:p2, p3:p4] = convolved
                # print(convolved.size, img[p1:p2, p3:p4].size)
                # for i in range(len((sus_area))):``
                #     print(sus_area[i])
                # p4 right edgs
                # p3 left edge
                # p2 bottom edge
                # p1 upper edge

            height, width, layer = img.shape
            size = (width, height)
            img_array.append(img)
            cv2.imshow("Frame", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    print(len(img_array))
    # Closes all the frames
    cv2.destroyAllWindows()
    out = cv2.VideoWriter(
        "videos/processed_videos/haarcascade_video_processed" + the_fileName[-6:],
        cv2.VideoWriter_fourcc(*"XVID"),
        15,
        size,
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# Example usage
# path = "videos/test_videos/video_recorded_"
# test_no = "2"
# blurThis(path + test_no + ".mp4")
