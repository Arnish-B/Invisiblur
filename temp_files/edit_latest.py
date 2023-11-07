import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
img_array = []
# Define the blur types
blur_types = ["gaussian", "median", "mosaic"]

sk_array = list()
ku_array = list()
sd_array = list()
var_array = list()
width = 640
height = 480
size = (width, height)


# Initialize the video capture object
cap = cv2.VideoCapture("smol_video_test.mp4")
# print(cap)


# Define the video codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "video_processed.mp4", fourcc, 25.0, (int(cap.get(3)), int(cap.get(4)))
)
img_array = []
# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    # print(ret)
    if ret == True:

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through each face and apply blur
        for (x, y, w, h) in faces:
            face_roi = frame[y : y + h, x : x + w]

            # Calculate the skewness, kurtosis, standard deviation, and variance of the face region
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(face_roi_gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(face_roi_gray, cv2.CV_32F, 0, 1)
            gradient = cv2.magnitude(gx, gy)
            skewness = np.mean((gradient - np.mean(gradient)) ** 3) / np.power(
                np.var(gradient), 1.5
            )
            sk_array.append(skewness)
            kurtosis = (
                np.mean((gradient - np.mean(gradient)) ** 4)
                / np.power(np.var(gradient), 2)
                - 3
            )
            ku_array.append(kurtosis)
            std_dev = np.std(gradient)
            sd_array.append(std_dev)
            variance = np.var(gradient)
            var_array.append(variance)

            # Apply different types of blurs to the face region
            if "gaussian" in blur_types:
                face_roi_gaussian = cv2.GaussianBlur(face_roi, (51, 51), 0)
                frame[y : y + h, x : x + w] = face_roi_gaussian

            if "median" in blur_types:
                face_roi_median = cv2.medianBlur(face_roi, 51)
                frame[y : y + h, x : x + w] = face_roi_median

            if "mosaic" in blur_types:
                face_roi_mosaic = cv2.resize(
                    face_roi, (10, 10), interpolation=cv2.INTER_LINEAR
                )
                face_roi_mosaic = cv2.resize(
                    face_roi_mosaic, (w, h), interpolation=cv2.INTER_NEAREST
                )
                frame[y : y + h, x : x + w] = face_roi_mosaic

            # Draw a rectangle around the face region and display the skewness and kurtosis values
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Skewness: {skewness:.2f}",
                (x, max(y - 20, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Kurtosis: {kurtosis:.2f}",
                (x, max(y - 50, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Std Dev: {std_dev:.2f}",
                (x, max(y - 80, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Variance: {variance:.2f}",
                (x, max(y - 110, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            height, width, layer = frame.shape
            size = (width, height)
            img_array.append(frame)
            # cv2.imshow("frame",frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #         break
    else:
        break
cv2.destroyAllWindows()
out = cv2.VideoWriter(
    "video_edit_mosaic_properties.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print(len(sk_array), len(ku_array), len(sd_array), len(var_array))
avg_skew = sum(sk_array) / len(sk_array)
avg_kurt = sum(ku_array) / len(ku_array)
avg_sd = sum(sd_array) / len(sd_array)
avg_var = sum(var_array) / len(var_array)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot the first bar graph in the top-left subplot
axs[0, 0].bar("Skewness", avg_skew)
axs[0, 0].set_title("Skewness")

# Plot the second bar graph in the top-right subplot
axs[0, 1].bar("Kurtosis", avg_kurt)
axs[0, 1].set_title("Kurtosis")

# Plot the third bar graph in the bottom-left subplot
axs[1, 0].bar("Standard Deviation", avg_sd)
axs[1, 0].set_title("Standard Deviation")

# Plot the fourth bar graph in the bottom-right subplot
axs[1, 1].bar("Variance", avg_var)
axs[1, 1].set_title("Variance")

# Adjust the layout of the subplots
plt.tight_layout()
plt.show()
