from deepface import DeepFace
import cv2
import numpy as np
import math


def blurThis(the_fileName):
    def sigmoid(x):
        return abs((1 / (1 + math.exp(-x))) - 0.5) / 10

    cap = cv2.VideoCapture(the_fileName)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    img_array = []
    count = 0
    original = []
    threshold = 8

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if count == 0:
                height, width, _ = img.shape
                count = 1
                original = img.copy()

            detected_face = DeepFace.extract_faces(
                img, detector_backend="mtcnn", enforce_detection=False
            )

            if detected_face is not None and "box" in detected_face:
                x, y, w, h = detected_face["box"]
                p1 = max(y - 20, 0)
                p2 = min(y + h + 20, height)
                p3 = max(x - 20, 0)
                p4 = min(x + w + 20, width)

                kernel = np.ones((15, 15), dtype=np.float32) / 225.0

                subframe = img[p1:p2, p3:p4]
                gray = cv2.cvtColor(subframe, cv2.COLOR_BGR2GRAY)
                convolved = cv2.filter2D(
                    gray, -1, kernel, borderType=cv2.BORDER_REPLICATE
                )
                convolved = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)

                original_sub_section = original[p1:p2, p3:p4]
                diff = np.abs(convolved - original_sub_section)
                mask = diff > threshold
                convolved[mask] = original_sub_section[mask]
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
        "videos/processed_videos/deepface_video_processed" + the_fileName[-6:],
        cv2.VideoWriter_fourcc(*"XVID"),
        15,
        size,
    )

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# Example usage
# path = "videos/test_videos/video_recorded_"
# test_no = "1"
# blurThis(path + test_no + ".mp4")
