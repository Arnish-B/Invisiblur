import cv2

# Load the video and create a VideoCapture object
video = cv2.VideoCapture("video.mp4")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define a list to store the previous five frames
previous_frames = []
img_array = []

# Define a variable to keep track of the current frame number
frame_num = 0

# Loop through each frame of the video
while True:
    # Read the current frame
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the current frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop through each face in the current frame
    for (x, y, w, h) in faces:
        # Check for movement within the face's grid
        if len(previous_frames) >= 5:
            previous_grid = previous_frames[-5][y:y+h, x:x+w]
            current_grid = gray[y:y+h, x:x+w]
            difference = cv2.absdiff(current_grid, previous_grid)
            if difference.mean() > 10:
                # Replace the current grid with the grid from 5 frames ago
                gray[y:y+h, x:x+w] = previous_frames[-5][y:y+h, x:x+w]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the current frame
    # cv2.imshow('Video', frame)

    # Add the current frame to the list of previous frames
    previous_frames.append(gray)

    # If the list of previous frames is longer than 5 frames, remove the oldest frame
    if len(previous_frames) > 5:
        previous_frames.pop(0)

    # Increment the frame number
    frame_num += 1

    height, width, layers = frame.shape
    size = (width, height)
    img_array.append(frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cv2.destroyAllWindows()
out = cv2.VideoWriter(
    "video_test_processed.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

############################################################################################################################################

# Load the video and create a VideoCapture object
video = cv2.VideoCapture("video.mp4")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
