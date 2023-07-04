import numpy as np
import streamlit as st
from PIL import Image


IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]


def get_upload_file(title="Choose a file"):
    uploaded_file = st.file_uploader(title)
    file_type = 'null'
    file_data = []
    status = False
    if uploaded_file is not None:
        status = True
        name_file = uploaded_file.name.split(".")[-1].lower()
        if name_file in IMG_FORMATS:
            file_type = 'image'
            file_data = np.array(Image.open(uploaded_file))
        elif name_file in VID_FORMATS:
            file_type = 'video'
    return status, file_type, file_data

