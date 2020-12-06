# -*- coding: utf-8 -*-
"""Copy of sample streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iAM7zS42S5uFmZ6Zl5QxL3AYPPx7YH9b
"""

# Importing Libraries
import streamlit as st
import io
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Title and Description
st.title('How Neural Networks see')
st.write("Just Upload your Image and Get Predictions on 1000 classes")
st.write("")


gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Loading Model
model = tf.keras.models.load_model("model.h5")

# Upload the image
uploaded_file = st.file_uploader("Choose a Image file", type=["png", "jpg", 'jpeg'])




# make a model to get output before flatten
activation_layer = model.get_layer(index=174)

while True:
  img = image.load_img(uploaded_file), target_size=(224, 224))
  x = preprocess_input(np.expand_dims(img, 0))
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

  ans = input("Continue? (Y/n)")
  if ans and ans[0].lower() == 'n':
    break