import streamlit as st
import pandas as pd
import keras
import csv
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


def get_image(image_path):
    # return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    # return cv2.imread(image_path)
    return cv2.resize(cv2.imread(image_path), dsize=(150, 150))
    # return preprocess_input(image.img_to_array(image.load_img(image_path, target_size=(223, 224, 3))))


def crop_image(b, draw):
    # print(b)
    x = b[0]
    y = b[1]
    w = b[2]
    h = b[3]
    crop_image = draw[y:h, x:w]
    return crop_image


def view_chart(performance, people, chart):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(people))
    ax.barh(y_pos, performance, align='center', color=['green', 'yellowgreen', 'dodgerblue','orange'])
    for index, value in enumerate(performance):
        plt.text(value, index, str(value))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()
    ax.set_xlabel('Number')
    ax.set_title(chart)
    plt.xlim(0, max(performance) + 400)
    plt.show()
    pass


def load_data_input_images(file_csv):
	data_images = []
	data_csv = np.array(pd.read_csv(file_csv, usecols=[0]))
	for data in data_csv:
		images = get_image(data[0])
		data_images.append(images)
	return np.array(data_images)

def process_dectection_human_sperm(model_path, video_file_input, output_detection_csv):
	model = models.load_model(model_path)
	#print(model.summary())

	# load label to names mapping for visualization purposes
	labels_to_names = {0: 'Sperm'}

	video_file = video_file_input
	# In this block you will import the address of your video
	save_path='output/frames' #path to the folder to save video
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
	data_sperm_images = []
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

		cnt_sperm = 0
		
		# visualize detections
		for box, score, label in zip(boxes[0], scores[0], labels[0]):
			# scores are sorted so we can break
			if score<0.5:
				break	
			
			cnt_sperm += 1
			color = (255,0,0)
			b = box.astype(int)

			image_sperm_crop = crop_image(b, draw)
			url_frame = frame_data.split('.')
			try:
				os.mkdir(url_frame[0])
			except:
				pass
			path = url_frame[0] + '/sperm_' + str(cnt_sperm) + '.jpg'

			cv2.imwrite(path, image_sperm_crop)

			# draw_box(draw, b, color=color)
			# caption = "{} {:.3f}".format(labels_to_names[label], score)
			# draw_caption(draw, b, caption)
			data.append([path, b[0], b[1], b[2], b[3], score, 'sperm']) # add the data to the list
			
			# image_sperm_crop = np.array(image_sperm_crop).reshape(30, 20, 3)
			# print(image_sperm_crop.shape)
			# data_sperm_images.append(image_sperm_crop)

		# cv2_imshow(draw)
		# cv2.imwrite('detected.jpg',draw)

	with open(output_detection_csv,'w',newline='') as f: # write the data to a csv file which will be used for tracking
		csvwriter=csv.writer(f)
		for row in data:
			csvwriter.writerow(row)
	pass


def process_classification_human_sperm(data_input, model_path):
	model=tf.keras.models.load_model(model_path)
	print(model.summary())

	y_predict = model.predict(data_input)
	y_target = np.argmax(y_predict, axis=1)

	view_chart([sum(y_target == 0), sum(y_target == 1), sum(y_target == 2)], ["Abnormal_Sperm", "Non-Sperm", "Normal_Sperm"], 'Chart Output')
	pass


# st.header('AI diagnosis human sperm using video')
# uploaded_files = st.file_uploader("Choose a video file (*.avi)", accept_multiple_files=True)

path_model_dectetion = 'sperm_detection_retinanet/model/final_retinanet_sperm_detection_3frames.h5'
path_model_classifier = 'sperm_classification/model/smids_mobiv2.h5'

video_file = 'sperm_detection_retinanet/RetinaNet_Motile_objects_Detection/13910927_4.avi'
csv_file = 'output/dectection_sperm.csv'

# process_dectection_human_sperm(path_model_dectetion, video_file, csv_file)

data_sperm_input = load_data_input_images(csv_file)
print(data_sperm_input.shape)

process_classification_human_sperm(data_sperm_input, path_model_classifier)