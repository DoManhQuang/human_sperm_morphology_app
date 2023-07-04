from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from datetime import datetime
import json
import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import my_yolov6

# Khởi tạo Flask Server Backend
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'message': "hello world"}

yolov6_model = my_yolov6.my_yolov6("./source/YOLOv6_Deploy/YOLOv6/weights/1.0/yolov6t_segrelu.pt", "cpu", 
                                   "./source/YOLOv6_Deploy/YOLOv6/weights/miamia-sperm.yaml", 640, False)

class PredictYoloV6(Resource):
    def post():
        data = request.get_json()
        img_in = data["data"]
        mode = data["mode"]
        if mode == "image":
            # Nhận diên qua model Yolov6
            _len, img_out, imgs_crop = yolov6_model.infer(img_in)
            return{
                "num_obj": _len,
                "img_out": img_out,
                "imgs_crop": imgs_crop
            }
        elif mode == "video":
            return {
                "message": "chua ho tro"
            }
        return{
            "message": "mode is image or video"
        }


api.add_resource(HelloWorld, '/')
api.add_resource(PredictYoloV6, '/api/yolo/v6/predict')

# Start Backend
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-port", type=str, default="6868", help="port")
    args = vars(parser.parse_args())
    app_port = args["port"]
    app.run(debug=True, host='0.0.0.0', port=app_port)