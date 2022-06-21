import keras
import csv
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
# from google.colab.patches import cv2_imshow
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def save_crop_sperms(b, draw, sperm_name, crop_path):
    print(b)
    x = b[0]
    y = b[1]
    w = b[2]
    h = b[3]
    crop_image = draw[y:h, x:w] 
    # cv2.imshow("Cropped Sperm", crop_image)
    cv2.imwrite(crop_path, crop_image)
    pass


model_path = 'model/final_retinanet_sperm_detection_3frames.h5' #Path to inference model

# load retinanet model

model = models.load_model(model_path)
print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Sperm'}

img1 = read_image_bgr('RetinaNet_Motile_objects_Detection/previous_frame.jpg')  # Load Previous Frame
img2 = read_image_bgr('RetinaNet_Motile_objects_Detection/Current_frame.jpg')  # Load Current Frame
img3 = read_image_bgr('RetinaNet_Motile_objects_Detection/next_frame.jpg')  # Load next Frame
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # convert to gray scale
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # convert to gray scale
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)  # convert to gray scale
image=np.concatenate((np.expand_dims(img1, axis=2), np.expand_dims(img2, axis=2), np.expand_dims(img3, axis=2)), axis=2) # concatenate 3 consecutive frames

print(image.shape)

draw = read_image_bgr('RetinaNet_Motile_objects_Detection/Current_frame.jpg')  # the original current frame image

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

image_width = draw.shape[1]
image_height = draw.shape[0]

# visualize detections
sperm_cnt = 0
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score<0.5:
        break
    sperm_cnt += 1

    color = (255, 0, 0)
    b = box.astype(int)

    sperm_name = "sperm_" + str(sperm_cnt)
    crop_path = "results/objects/" + sperm_name + "_.jpg"
    save_crop_sperms(b, draw, sperm_name, crop_path)
 
    # draw_box(draw, b, color=color)
    # caption = "{} {:.3f}".format(labels_to_name s[label], score)
    # draw_caption(draw, b, caption)

# cv2.imshow("image", draw[1])
# name = 'results/detected_image_sperm_'+ str(sperm_cnt) +'_.jpg'
# cv2.imwrite(name, draw)
