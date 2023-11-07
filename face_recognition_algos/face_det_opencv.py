import cv2
import numpy as np
import math


def blurThis(the_fileName):
    def sigmoid(x):
        return abs((1 / (1 + math.exp(-x))) - 0.5) / 10

    cap = cv2.VideoCapture(the_fileName)
    # net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pbtxt", "opencv_face_detector.pbtxt")
    net = cv2.dnn.readNetFromTensorflow(
        "face_recognition_algos/opencv_face_detector_uint8.pb",
        "face_recognition_algos/opencv_face_detector.pbtxt",
    )

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    img_array = []
    count = 0
    original = []
    threshold = 20

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if count == 0:
                height, width, _ = img.shape
                count = 1
                original = img.copy()
                nonBlurred_original = original.copy()

            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 117, 123)
            )
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    # a = detections[0, 0, i, :]
                    box = detections[0, 0, i, 3:7] * \
                        np.array([width, height, width, height])
                    (x, y, w, h) = box.astype(int)

                    p1 = max(y, 0)
                    p2 = min(h, height)
                    p3 = max(x, 0)
                    p4 = min(w, width)

                    kernel = np.ones((15, 15), dtype=np.float32) / 225.0

                    subframe = img[p1:p2, p3:p4]
                    gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
                    convolved = cv2.filter2D(
                        gray, -1, kernel, borderType=cv2.BORDER_REPLICATE
                    )
                    convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)

                    original_sub_section = original[p1:p2, p3:p4]
                    nonBlurred_original_subSection = nonBlurred_original[p1:p2, p3:p4]
                    diff = np.abs(convolved - original_sub_section)
                    mask = diff > threshold
                    convolved[mask] = nonBlurred_original_subSection[mask]
                    img[p1:p2, p3:p4] = convolved

            height, width, _ = img.shape
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
        "videos/processed_videos/SSD_video_processed" + the_fileName[-6:],
        cv2.VideoWriter_fourcc(*"XVID"),
        15,
        size,
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# # Example usage
# path = "videos/test_videos/video_recorded_"

# test_no = "2"
# blurThis(path + test_no + ".mp4")
