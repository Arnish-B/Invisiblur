import cv2
import numpy as np

cap = cv2.VideoCapture("smol_video_test.mp4")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# background_subtractor = cv2.createBackgroundSubtractorMOG2(
#     history=5, varThreshold=50)

if cap.isOpened() == False:
    print("Error opening video stream or file")

frame_count = 0
face_grid = []
img_array = []
prev_frames = []
prev_face = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Store previous frames
    if len(prev_frames) < 5:
        prev_frames.append(gray)
    else:
        prev_frames.pop(0)
        prev_frames.append(gray)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through the faces
    for (x, y, w, h) in faces:
        # Get the grid of the face
        face_grid = gray[y:y+h, x:x+w]

        # Check if there is a movement in the grid
        if prev_face is not None and prev_face.shape == face_grid.shape and cv2.absdiff(prev_face, face_grid).mean() > 5:
            print("inside calculation")
            # Replace the grid with the corresponding grid from the previous frame
            gray[y:y+h, x:x+w] = prev_face
        else:
            # Update prev_face with the current face
            prev_face = face_grid

        # Draw a green square around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow('frame', gray)

    height, width, layers = frame.shape
    size = (width, height)
    img_array.append(gray)

    if cv2.waitKey(1) == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()
out = cv2.VideoWriter(
    "video_test_processed.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# import cv2
# import numpy as np

# img_array = []

# cap = cv2.VideoCapture("video.mp4")
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# prev_frames = []
# prev_face = None

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Store previous frames
#     if len(prev_frames) < 5:
#         prev_frames.append(gray)
#     else:
#         prev_frames.pop(0)
#         prev_frames.append(gray)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     # Loop through the faces
#     for (x, y, w, h) in faces:
#         # Get the grid of the face
#         face_grid = gray[y:y+h, x:x+w]

#         # Check if there is a movement in the grid
#         if prev_face is not None and prev_face.shape == face_grid.shape and cv2.absdiff(prev_face, face_grid).mean() > 2:
#             # Replace the grid with the corresponding grid from the previous frame
#             gray[y:y+h, x:x+w] = prev_face
#         else:
#             # Update prev_face with the current face
#             prev_face = face_grid

#         # Draw a green square around the face
#         # cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     # cv2.imshow('frame', gray)

#     height, width, layers = frame.shape
#     size = (width, height)
#     img_array.append(frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# out = cv2.VideoWriter(
#     "video_test_processed.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
# )


# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
