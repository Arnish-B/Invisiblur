import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img_array = []
while True:
    ret, img = cap.read()
    cv2.imshow("test", img)
    height, width, layer = img.shape
    size = (width, height)
    img_array.append(img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
print(len(img_array))
# Closes all the frames
cv2.destroyAllWindows()
out = cv2.VideoWriter(
    "videos/recorder_video.mp4",
    cv2.VideoWriter_fourcc(*"DIVX"),
    15,
    size,
)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
