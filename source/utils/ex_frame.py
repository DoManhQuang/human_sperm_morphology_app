import os
import cv2
from tqdm import tqdm
import numpy as np
import sys
ROOT = os.getcwd()
if str(ROOT) == "/":
    ROOT = "/code"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print("ROOT : ", ROOT)


def save_image(video_cap, file_name, time_msec):
    video_cap.set(cv2.CAP_PROP_POS_MSEC, time_msec)
    ret, frame = video_cap.read()
    if ret:
        # print(filename)
        cv2.imwrite(file_name, frame)
    pass


def get_duration_in_seconds(video_cap):
    return video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_cap.get(cv2.CAP_PROP_FPS)


def extract_video(path_dir, video_in, num_frame=5):
    if not video_in.isOpened():
        print("\nError opening video file")

    minutes = 0
    seconds = get_duration_in_seconds(video_cap=video_in)
    param = np.linspace(0.0, 1.0, num=num_frame)
    print(param)
    for par in tqdm(range(0, len(param))):
        if video_in.isOpened():
            t_msec = 1000 * (minutes * 60 + seconds * param[par])
            filename = os.path.join(path_dir, f"IMG_{par}.jpg")
            save_image(video_cap=video_in, file_name=filename, time_msec=t_msec)

    # if len(os.listdir(path_dir)) == len(param):
    #     return "DONE!!"
    # return "ERROR Ex-Frame"
    return "DONE"
