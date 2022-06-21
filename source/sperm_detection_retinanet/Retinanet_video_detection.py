import keras
import csv
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

model_path = 'model/final_retinanet_sperm_detection_3frames.h5' #Path to inference model

# load retinanet model

model = models.load_model(model_path)
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Sperm'}

# In this block you will import the address of your video
video_file='RetinaNet_Motile_objects_Detection/13910927_4.avi' #path to the video file
save_path='frames' #path to the folder to save video
video_data=[]
cap = cv2.VideoCapture(video_file) #read video
count = 0
import shutil
try:
  shutil.rmtree(save_path)
except:
  pass
#create folder
try:
    os.mkdir(save_path)
except:
    pass
while cap.isOpened():
    ret,frame = cap.read()
    if ret is True:     
        count = count + 1
        cv2.imwrite("{}/{}.jpg".format(save_path,count), frame) #write frame
        video_data.append("{}/{}.jpg".format(save_path,count))  #add the data to the list
    else:
        break
cap.release()

data=[]
for index,frame_data in enumerate(video_data):
  if index==0: #first frame
    img1 = read_image_bgr(frame_data) #Load Previous Frame
    img2 = read_image_bgr(frame_data) #Load Current Frame
    img3 = read_image_bgr(video_data[index+1]) #Load next Frame
  elif index== len(video_data)-1: #last frame
    img1 = read_image_bgr(video_data[index-1]) #Load Previous Frame
    img2 = read_image_bgr(frame_data) #Load Current Frame
    img3 = read_image_bgr(frame_data) #Load next Frame
  else: #other frames
    img1 = read_image_bgr(video_data[index-1]) #Load Previous Frame
    img2 = read_image_bgr(frame_data) #Load Current Frame
    img3 = read_image_bgr(video_data[index+1]) #Load next Frame

  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #convert to gray scale
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #convert to gray scale
  img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) #convert to gray scale
  image=np.concatenate((np.expand_dims(img1,axis=2),np.expand_dims(img2,axis=2),np.expand_dims(img3,axis=2)),axis=2) #concatenate 3 consecutive frames

  draw = read_image_bgr(frame_data) #the original current frame image

  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)

  # process image
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  print("processing time: ", time.time() - start)

  # correct for image scale
  boxes /= scale


  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score<0.5:
        break
  
      color = (255,0,0)
      
      b = box.astype(int)
      draw_box(draw, b, color=color)
      
      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)
      data.append([frame_data,b[0],b[1],b[2],b[3],'sperm']) #add the data to the list

  # cv2_imshow(draw)
  cv2.imwrite('detected.jpg',draw)

with open('detections.csv','w',newline='') as f: #write the data to a csv file which will be used for tracking
  csvwriter=csv.writer(f)
  for row in data:
    csvwriter.writerow(row)

#Download detections.csv and use it for perform tracking (via modified csr-dcf.py file)