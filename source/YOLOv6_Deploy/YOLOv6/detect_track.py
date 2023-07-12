
import os
print( os.getcwd())
# Cấu hình chung
#file wight của yolo
checkpoint:str ="./weights/1.0/last_ckpt" 
#Loại thiết bị
device:str = "cpu"#@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}
max_cosine_distance = 0.4
nn_budget = None
#Weigh của deepsort
model_filename = 'model_data/mars-small128.pb'

#Yolo import
import os, requests, torch, math, cv2
import numpy as np
import PIL
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from typing import List, Optional
import my_yolov6
from yolov6.layers.common import DetectBackend


#Deep sort import
from deep_sort.tools import generate_detections as gdet
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

model = DetectBackend(f"./{checkpoint}.pt", device=device)


# model = my_yolov6.my_yolov6("./weights/1.0/last_ckpt.pt", "cpu", 
#                                    "./weights/dataset.yaml", 640, False)
stride = model.stride
#class_names = load_yaml("./data/coco.yaml")['names']
# lass_names=["S", "I"]c
# if half & (device.type != 'cpu'):
#   model.model.half()
# else:
#   model.model.float()
#   half = False

class Node:
    def __init__(self,_x,_y,_id) -> None:
        self.x=_x,
        self.y=_y,
        self.len = 0
        self.id=_id
    def update(self,_x,_y):
        self.len = abs(_x-self.x) + abs(_y-self.y)
        self.x=_x,
        self.y=_y


def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def precess_image(rgb_img, img_size, stride, half):
  '''Process image before image inference.'''
  try:
    assert rgb_img is not None, f'Invalid image'
  except Exception as e:
    LOGGER.Warning(e)
  image = letterbox(rgb_img, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, rgb_img
def node_to_json(obj):
    if isinstance(obj, Node):
        return json.dumps({obj.id,obj.len})
def node_to_dict(obj):
    if isinstance(obj, Node):
        return float(obj.len)
import json
class DeepSortTracker:
  def __init__(self) -> None:
      self.memo = {}
      #Khởi tạo deep_sort
      self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
      self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
      self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
      self.tracker = Tracker(self.metric)
  def detect_per_frame(self,rgb_img, img_size=640,conf_thres=.24,iou_thres=.45,max_det=1000,agnostic_nms=False):
    img_size = check_img_size(img_size, s=stride)
    img, img_src = precess_image(rgb_img, img_size, stride, half)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes:Optional[List[int]] = None # the classes to keep
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_src.copy()
    
    if len(det):
      det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
      bboxes = []
      scores = []
      for *xyxy, conf, cls in reversed(det):
          bbox_left = min([xyxy[0].item(), xyxy[2].item()])
          bbox_top = min([xyxy[1].item(), xyxy[3].item()])
          bbox_w = abs(xyxy[0].item() - xyxy[2].item())
          bbox_h = abs(xyxy[1].item() - xyxy[3].item())
          box = [bbox_left, bbox_top, bbox_w, bbox_h]
          if bbox_w <=0 or bbox_h<=0:
            print(xyxy)
            continue
          bboxes.append(box)
          scores.append(conf.item())
          
        
      if len(bboxes)<1:
        print('No object')
        return img_ori
      # DeepSORT -> Getting appearence features of the object.
      features = self.encoder(img_src, bboxes)
      # DeepSORT -> Storing all the required info in a list.
      detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(bboxes, scores, features)]

      # DeepSORT -> Predicting Tracks. 
      self.tracker.predict()
      self.tracker.update(detections)

      for track in self.tracker.tracks:
        if not track.is_confirmed():
            continue
        bbox = list(track.to_tlbr())
        txt = 'id:' + str(track.track_id)
        if self.memo.get(track.track_id) is not None:
          self.memo.get(track.track_id).update(bbox[0],bbox[1])
        else:
          self.memo[track.track_id]= Node(bbox[0] ,bbox[1],track.track_id)
      return self.memo