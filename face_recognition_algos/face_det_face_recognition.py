import cv2
import numpy as np
import math
import face_recognition


def blurThis(the_fileName):
    def sigmoid(x):
        return abs((1 / (1 + math.exp(-x))) - 0.5) / 10

    cap = cv2.VideoCapture(the_fileName)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    img_array = []
    size= 0
    count = 0
    original = []
    threshold = 8
    a = 15
    kernel = np.ones((a, a), dtype=np.float32) / (a**2)
    flag = 0
    f = 0

    while cap.isOpened():
        f += 1
        ret, img = cap.read()

        if ret == True:
            if count == 0:
                height, width, layer = img.shape
                count = 1
                original = img.copy()
                nonBlured_original = original.copy()
                grid = [[1 for i in range(6)] for i in range(6)]

                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                convolved = cv2.filter2D(
                    gray, -1, kernel, borderType=cv2.BORDER_REPLICATE
                )
                convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
                height, width = original.shape[:2]
                original = convolved[:height, :width]

            face_locations = face_recognition.face_locations(img)

            for top, right, bottom, left in face_locations:
                p1 = max(top, 0)
                p2 = min(bottom, height)
                p3 = max(left, 0)
                p4 = min(right, width)

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

            height, width, layer = img.shape
            size = (width, height)
            img_array.append(img)
            cv2.imshow("Frame", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    print(len(img_array))
    cv2.destroyAllWindows()
    out = cv2.VideoWriter(
        "videos/processed_videos/face_recogintion_video_processed"
        + the_fileName[-6:],
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
