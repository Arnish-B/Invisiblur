import cv2
from flask import request, render_template
from tqdm.gui import trange
from application import app


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/", methods=["POST", "GET"])
def show():
    if request.method == "POST":
        name2 = request.form["vid"]
        blur_type = request.form["blur_type"]

        # Load Haar cascade classifier and open video file as before
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cap = cv2.VideoCapture(name2)
        img_array = []

        for f in trange(
            500, desc="Processing", bar_format="{desc}: {percentage:3.0f}%"
        ):
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                ROI = img[y : y + h, x : x + w]
                if blur_type == "invisi-blur":
                    print(ROI)
                elif blur_type == "median":
                    blur = cv2.medianBlur(ROI, 27)
                elif blur_type == "gaussian":
                    blur = cv2.GaussianBlur(ROI, (27, 27), 0)
                else:
                    # Apply mosaic filter
                    kernel_size = min(w, h) // 20
                    mosaic_size = max(w, h) // kernel_size

                    # Resize the ROI to the mosaic size
                    resized_ROI = cv2.resize(
                        ROI, (mosaic_size, mosaic_size), interpolation=cv2.INTER_AREA
                    )

                    # Resize the mosaic back to the original size
                    mosaic = cv2.resize(
                        resized_ROI, (w, h), interpolation=cv2.INTER_NEAREST
                    )

                    # Apply the mosaic filter to the ROI
                    blur = mosaic

                img[y : y + h, x : x + w] = blur

                height, width, layers = img.shape
                size = (width, height)

            img_array.append(img)

        # Release video capture object and create new video file as before

        out = cv2.VideoWriter(
            "video_processed.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 15, size
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        return render_template(
            "index.html",
            info="Anonymized video successfully saved in the folder containing the original video.",
        )
    else:
        return render_template("index.html")
