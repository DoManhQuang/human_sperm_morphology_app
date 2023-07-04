import pickle
from flask import Flask, render_template, request, jsonify
import os
from random import random
import my_yolov6
import cv2
# import streamlit as st


# def uploaded_video(title="Choose a file", display_video=True):
#     uploaded_file = st.file_uploader(title)
#     bytes_data = None
#     if uploaded_file is not None:
#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         # st.write(bytes_data)
#         if display_video:
#             st.video(bytes_data)
#     return bytes_data


yolov6_model = my_yolov6.my_yolov6("./source/YOLOv6_Deploy/YOLOv6/weights/1.0/yolov6t_segrelu.pt", "cpu", 
                                   "./source/YOLOv6_Deploy/YOLOv6/weights/miamia-sperm.yaml", 640, False)

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                ndet, frame, imgs_crop = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)

                if ndet!=0:
                    cv2.imwrite(path_to_save, frame)

                    # Trả về kết quả
                    return render_template("index.html", user_image=image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet)
                else:
                    return render_template('index.html', msg='Không nhận diện được vật thể')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


@app.route('/hello', methods=['GET', 'POST'])
def hello():
    return jsonify({"result": "hello world"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)