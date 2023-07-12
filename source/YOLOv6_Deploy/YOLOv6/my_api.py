from flask import Flask, request
from flask_restful import Resource, Api
import tensorflow as tf
import os, sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import my_yolov6
from PIL import Image
import numpy as np
from detect_track import DeepSortTracker
# load model

yolov6_model = my_yolov6.my_yolov6("./weights/1.0/last_ckpt.pt", "cpu", 
                                   "./weights/dataset.yaml", 640, False)

sperm_cls  = None #tf.keras.models.load_model('./source/sperm_classification/model/smids_mobiv2.h5')

# Khởi tạo Flask Server Backend
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'message': "hello world"}


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


class CNNPredictSperm(Resource):
    def post():
        data = request.get_json()
        img_in = data["data"]
        results = sperm_cls.predict(img_in)
        return{
            "results": results
        }

class Tracking(Resource):
    def post(self):
        images = request.files.getlist('images')
        tracker =  DeepSortTracker()
        for image in images:
            img = Image.open(image)
            img_array = np.array(img)
            tracker.detect_per_frame(img_array)
        print(tracker.memo)
        return '!Upload success'


api.add_resource(HelloWorld, '/')
api.add_resource(PredictYoloV6, '/api/yolo/v6/predict')
api.add_resource(CNNPredictSperm, '/api/cnn/cls/predict')
api.add_resource(Tracking,'/api/yolo/v6/tracking')
# Start Backend
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-port", type=str, default="6868", help="port")
    args = vars(parser.parse_args())
    app_port = args["port"]
    app.run(debug=True, host='0.0.0.0', port=app_port)