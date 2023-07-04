import streamlit as st
from utils.utils import uploaded_video
from source.utils.yolov6_detection import my_yolov6
import cv2


# yolo_v6_model = my_yolov6("../source/weights/1.0/yolov6t_segrelu.pt", "cpu", "../source/weights/miamia-sperm.yaml", 640, False)

# frame = cv2.imread("source/video/S_0001_0001.png")
# frame, no_object = yolo_v6_model.infer(frame)
# print(frame)


if __name__ == '__main__':
    st.title("CHUẨN ĐOÁN SỨC KHỎE SINH SẢN NAM GIỚI")
    with st.form("display_video_form"):
        st.write("Hiển Thị Video Đầu Vào")
        bytes_video = uploaded_video(display_video=False, title="Lựa chọn video đầu vào")
        submitted = st.form_submit_button("Bắt Đầu Xử Lý")
        if submitted:
            st.write("Kết Quả")

            # hien thi video
            # st.video(bytes_video)

            # xu ly detection

            # xy ly classification

        pass


    st.write("Copyright © 2023, Do Manh Quang.")

