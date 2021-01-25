# Importing Libraries
import streamlit as st
import io
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Title and Description
st.title('How Neural Networks see')
st.write("Just Upload your Image and Get Predictions on 1000 classes")
st.write("")



# Loading Model
model = tf.keras.models.load_model("model.h5")

# Upload the image
uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg", 'jpeg'])

if uploaded_file is True:

	imgs = plt.imread(io.BytesIO(uploaded_file.read()))

	img = np.array(imgs.resize((224, 224)))
	x = preprocess_input(img)
	fmaps = model.predict(x)[0] # 7 x 7 x 2048

	# get predicted class
	probs = model.predict(x)
	classnames = decode_predictions(probs)[0]
	print(classnames)
	classname = classnames[0][1]
	pred = np.argmax(probs[0])

	# get the 2048 weights for the relevant class
	w = W[:, pred]

	# "dot" w with fmaps
	cam = fmaps.dot(w)

	# upsample to 224 x 224
	# 7 x 32 = 224
	cam = sp.ndimage.zoom(cam, (32, 32), order=1)

	plt.subplot(1,2,1)
	plt.imshow(img, alpha=0.8)
	plt.axis('off')
	plt.imshow(cam, cmap='jet', alpha=0.5)
	plt.subplot(1,2,2)
	plt.imshow(img)
	plt.axis('off')
	plt.title(classname)
	plt.show()
