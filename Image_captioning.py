#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import re
import cv2
import json
import pickle,keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
plt.style.use('seaborn')

model = ResNet50(weights="imagenet",input_shape=(224,224,3))
new_model = Model(model.input,model.layers[-2].output)

def get_image_encodings(img):
    feature_vector = new_model.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector

def preprocessing(img_path):
    img = image.load_img(img_path,target_size=(224,224))
    img = image.img_to_array(img)
    img = img.reshape((1,224,224,3))
    img = preprocess_input(img)
    return get_image_encodings(img)

with open("word_to_idx.pkl","rb") as f:
    word_to_idx=pickle.load(f)
with open("idx_to_word.pkl","rb") as f:
    idx_to_word=pickle.load(f)

max_len = 37

model=load_model('model_weights/model_final.h5')

def predict_caption(img_path):
    photo=preprocessing(img_path)
    in_text = '<start>'
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len,padding='post')
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word[yhat]
        in_text += (' ' + word)
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


print(predict_caption('1.jpg'))






