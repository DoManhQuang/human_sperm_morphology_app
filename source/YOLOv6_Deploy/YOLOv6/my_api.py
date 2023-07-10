from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf
import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import my_yolov6

# load model

yolov6_model = my_yolov6.my_yolov6(r"J:\HaUI\nckh\human_sperm_morphology_app\source\YOLOv6_Deploy\YOLOv6\weights\1.0\yolov6t_segrelu.pt", "cpu",
                                   r"J:\HaUI\nckh\human_sperm_morphology_app\source\YOLOv6_Deploy\YOLOv6\weights\miamia-sperm.yaml", 640, False)

sperm_cls = tf.keras.models.load_model(r'J:\HaUI\nckh\human_sperm_morphology_app\source\sperm_classification\model\smids_mobiv2.h5')

# Khởi tạo Flask Server Backend
app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'message': "hello world"}


class PredictYoloV6(Resource):
    def post(self):
        data = request.get_json()
        img_in = data["data"]
        mode = data["mode"]
        if mode == "image":
            # Nhận diên qua model Yolov6
            _len, img_out, imgs_crop = yolov6_model.infer(img_in)
            return{
                "img_num_obj": _len,
                "img_out": img_out,
                "imgs_crop": imgs_crop
            }
        elif mode == "video":
            video_data=[]
            video_num_obj=[]
            video_crop=[]
            for img in img_in:
                _len, img_out, imgs_crop = yolov6_model.infer(img)
                video_data.append(img_out)
                video_num_obj.append(_len)
                video_crop.append(imgs_crop)
            return {
                "video_num_obj": video_num_obj,
                "video_out": video_data,
                "video_crop": video_crop
            }
        return{
            "message": "mode is image or video"
        }


class CNNPredictSperm(Resource):
    def post(self):
        data = request.get_json()
        img_in = data["data"]
        results = sperm_cls.predict(img_in)
        return{
            "results": results
        }


api.add_resource(HelloWorld, '/')
api.add_resource(PredictYoloV6, '/api/yolo/v6/predict')
api.add_resource(CNNPredictSperm, '/api/cnn/cls/predict')


# Start Backend
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-port", type=str, default="6969", help="port")
    args = vars(parser.parse_args())
    app_port = args["port"]
    app.run(debug=True, host='0.0.0.0', port=app_port)
