import cv2
from flask import request, render_template
from tqdm.gui import trange
from application import app
from face_recognition_algos import (
    face_det_harcascade,
    face_det_dlib,
    face_det_deepface,
    face_det_mtcnn,
    face_det_opencv,
    face_det_face_recognition,
)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/", methods=["POST", "GET"])
def show():
    if request.method == "POST":
        pp = "videos/test_videos/"
        file_path = pp+request.form["vid"]
        blur_type = request.form["face_detection_type"]

        if blur_type == "haarcascade":
            print(file_path)
            face_det_harcascade.blurThis(file_path)
            return render_template(
                "index.html",
                info="Anonymized video successfully saved in the folder containing the original video.",
            )
        elif blur_type == "mtcnn":
            print(file_path)
            face_det_mtcnn.blurThis(file_path)
            return render_template(
                "index.html",
                info="Anonymized video successfully saved in the folder containing the original video.",
            )
        elif blur_type == "dlib":
            print(file_path)
            face_det_dlib.blurThis(file_path)
            return render_template(
                "index.html",
                info="Anonymized video successfully saved in the folder containing the original video.",
            )
        elif blur_type == "OpenCV":
            print(file_path)
            face_det_opencv.blurThis(file_path)
            return render_template(
                "index.html",
                info="Anonymized video successfully saved in the folder containing the original video.",
            )

        elif blur_type == "FaceRecognition":
            print(file_path)
            face_det_face_recognition.blurThis(file_path)
            return render_template(
                "index.html",
                info="Anonymized video successfully saved in the folder containing the original video.",
            )

        return render_template(
            "index.html",
            info="Anonymized video successfully saved in the folder containing the original video.",
        )
    else:
        return render_template("index.html")
