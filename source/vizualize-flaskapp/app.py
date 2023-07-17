import os, sys
import cv2
from flask import (
    Flask,
    Response,
    render_template,
    request,
    session,
    redirect,
    send_file,
    url_for,
)

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from source.utils import ex_frame

# create folder static
if not os.path.exists("static"):
    os.mkdir("static")

if not os.path.exists("static/upload"):
    os.mkdir("static/upload")

if not os.path.exists("static/output"):
    os.mkdir("static/output")

app = Flask(__name__, template_folder="template", static_folder="static")
app.secret_key = "abc"
app.config["UPLOAD_FOLDER"] = "static/upload"
app.config["UPLOAD_IMG_EXT"] = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
    "mpo",
]
app.config["UPLOAD_VID_EXT"] = ["mp4", "mov", "avi", "mkv"]
app.config["OUTPUT_FOLDER"] = "static/output"

# FACE_DETECTORS = ["haar cascade", "retina face"]
# FAS_MODELS = ["large", "small", "large_rf-f12", "large_rf-f12-e2"]

YOLO_SERIES = ['YOLOv3', 'YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8']
OPT_ACTIVATION = ['relu', 'seg_relu', 'silu']

global cap, fd, fas, cam_on

cam_on = False
cap = None
fd = None
fas = None


def get_media_file(filename):
    return os.path.join(app.config["UPLOAD_FOLDER"], filename)


def is_image(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_IMG_EXT"]


def is_video(file_path):
    extension = file_path.split(".")[-1].lower()
    return extension in app.config["UPLOAD_VID_EXT"]


def render_upload(
    html="upload_file.html",
    iimg=None,
    oimg=None,
    ivideo=None,
    ovideo=None,
    yolo_series=YOLO_SERIES,
    opt_activation=OPT_ACTIVATION,
    selected_face_detector=YOLO_SERIES[0],
    selected_fas_model=OPT_ACTIVATION[0],
    fd_time=None,
    fas_time=None,
    noti=None
):
    return render_template(
        html,
        iimg=iimg,
        oimg=oimg,
        ivideo=ivideo,
        ovideo=ovideo,
        face_detectors=yolo_series,
        fas_models=opt_activation,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        fd_time=fd_time,
        fas_time=fas_time,
        noti=noti
    )


def render_camera(
    html="camera.html",
    yolo_series=YOLO_SERIES,
    opt_activation=OPT_ACTIVATION,
    selected_face_detector=YOLO_SERIES[0],
    selected_fas_model=OPT_ACTIVATION[0],
    noti=None
):
    global cam_on
    return render_template(
        html,
        cam_on=cam_on,
        face_detectors=yolo_series,
        fas_models=opt_activation,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        noti = noti
    )


def render_phonecamera(
    html="phone_camera.html",
    cam_ip=None,
    yolo_series=YOLO_SERIES,
    opt_activation=OPT_ACTIVATION,
    selected_face_detector=YOLO_SERIES[0],
    selected_fas_model=OPT_ACTIVATION[0],
    noti=None
):
    global cam_on
    return render_template(
        html,
        cam_on=cam_on,
        cam_ip=cam_ip,
        face_detectors=yolo_series,
        fas_models=opt_activation,
        selected_face_detector=selected_face_detector,
        selected_fas_model=selected_fas_model,
        noti=noti
    )


@app.route("/")
def index():
    session["fas_model"] = OPT_ACTIVATION[0]
    session["face_detector"] = YOLO_SERIES[0]
    return render_template(
        "home.html",
        face_detectors=YOLO_SERIES,
        fas_models=OPT_ACTIVATION,
    )


@app.route("/", methods=["POST"])
def goto():
    if request.form.get("upload") == "Upload":
        return redirect(url_for("upload"))
    elif request.form.get("camera") == "Camera":
        return redirect(url_for("camera"))
    elif request.form.get("mobile-phone-camera") == "Mobile phone camera":
        return redirect(url_for("phonecamera"))
    return redirect(url_for("index"))


@app.route("/back", methods=["GET"])
def backtohome():
    global cap, cam_on
    if cam_on:
        cap.release()
        cam_on = False
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        input_file = request.files["input_file"]

        if is_image(input_file.filename):
            path = get_media_file(input_file.filename)
            input_file.save(path)
            session["uploaded_img_path"] = path
            return render_upload(iimg=path)

        elif is_video(input_file.filename):
            path = get_media_file(input_file.filename)
            input_file.save(path)
            folder_in = input_file.filename.split(".")[0]
            if not os.path.exists(f"static/upload/{folder_in}"):
                os.mkdir(f"static/upload/{folder_in}")
            else:
                os.rmdir(f"static/upload/{folder_in}")
                os.mkdir(f"static/upload/{folder_in}")
            session["uploaded_img_path"] = path
            return render_upload(ivideo=path)

        else:
            return render_upload(noti="Please upload image or video file")

    return render_upload()


# @app.route("/camera", methods=["GET", "POST"])
# def camera():
#     global cap, cam_on, fas, fd
#     if request.method == "GET":
#         session["fas_model"] = FAS_MODELS[0]
#         session["face_detector"] = FACE_DETECTORS[0]
#         if cam_on:
#             cap.release()
#             cam_on = False
#         return render_camera()
#     else:
#         if request.form.get("start") == "Start":
#             if (not fas) or (session["fas_model"] != request.form.get("fas-model-btn")):
#                 session["fas_model"] = request.form.get("fas-model-btn")
#                 fas = fas_model(session["fas_model"])
#             if (not fd) or (
#                 session["face_detector"] != request.form.get("face-detector-btn")
#             ):
#                 session["face_detector"] = request.form.get("face-detector-btn")
#                 fd = face_detector(session["face_detector"])
#             cam_on = True
#             cap = cv2.VideoCapture(0)
#
#         elif request.form.get("stop") == "Stop":
#             cap.release()
#             cam_on = False
#         return render_camera(selected_face_detector=session["face_detector"],
#                              selected_fas_model=session["fas_model"])
#
#
# @app.route("/phonecamera", methods=["GET", "POST"])
# def phonecamera():
#     global cap, cam_on, fd, fas
#     if request.method == "GET":
#         session["fas_model"] = FAS_MODELS[0]
#         session["face_detector"] = FACE_DETECTORS[0]
#         if cam_on:
#             cap.release()
#             cam_on = False
#         return render_phonecamera()
#     else:
#         if request.form.get("start") == "Start":
#             if (not fas) or (session["fas_model"] != request.form.get("fas-model-btn")):
#                 session["fas_model"] = request.form.get("fas-model-btn")
#                 fas = fas_model(session["fas_model"])
#             if (not fd) or (
#                 session["face_detector"] != request.form.get("face-detector-btn")
#             ):
#                 session["face_detector"] = request.form.get("face-detector-btn")
#                 fd = face_detector(session["face_detector"])
#
#             cam_ip = request.form.get("cam_ip")
#             cap = cv2.VideoCapture("https://" + cam_ip + "/video")
#             cam_on = True
#             return render_phonecamera(cam_ip=cam_ip,
#                                       selected_face_detector=session["face_detector"],
#                                       selected_fas_model=session["fas_model"])
#
#         elif request.form.get("stop") == "Stop":
#             cap.release()
#             cam_on = False
#
#         return render_phonecamera(selected_face_detector=session["face_detector"],
#                                   selected_fas_model=session["fas_model"])
#
#
# @app.route("/stream", methods=["GET"])
# def stream():
#     return Response(
#         generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     )
#
#


@app.route("/submit", methods=["POST", "GET"])
def submit():
    video = cv2.VideoCapture(session["uploaded_img_path"])
    folder_in = session["uploaded_img_path"].split(".")[0]
    print("Foler IN : ", folder_in)
    if not video.isOpened():
        print("\nError opening video file ", session["uploaded_img_path"])

    status_ex = ex_frame.extract_video(path_dir=f"./source/vizualize-flaskapp/{folder_in}", video_in=video, num_frame=20)
    print("status ex frame: ", status_ex)
    return redirect("/")


@app.route("/download", methods=["GET"])
def download():
    return send_file(session["last_output_img"], as_attachment=True)


if __name__ == "__main__":
    app.run()
