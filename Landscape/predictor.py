import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import torch
import torchvision.transforms.functional as TF


size = (100,100)

def preprocess_img(image):

	if(image.mode == 'L'):
		image = image.convert('RGB')
	image = image.resize(size)
	return image

def image_loader(image):
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	return image

def get_classlabel(class_code):
    labels = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea' , 5:'street'}
    return labels[class_code]



class landscape_predictor:
	def __init__(self):
		self.model = keras.models.load_model('final_model.h5')
		self.model.load_weights('model_weights.h5')

	def predict(self, request):

		f=request.files['image']
		image = Image.open(f)
		image = preprocess_img(image)
		image = image_loader(image)
		num_prediction = np.argmax(self.model.predict(image))
		label = get_classlabel(num_prediction)
		return label



