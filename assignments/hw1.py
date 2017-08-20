import os, json
import numpy as np

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom


import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

FILES_PATH = 'http://www.platform.ai/models/'; CLASS_FILE='imagenet_class_index.json'

fpath = get_file(CLASS_FILE, FILES_PATH + CLASS_FILE, cache_subdir='models')
with open(fpath) as f: class_dict = json.load(f)
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

def ConvBlock(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(filters, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

def FGBlock(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]

def VGG_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

    ConvBlock(2, model, 64)
    ConvBlock(2, model, 128)
    ConvBlock(3, model, 256)
    ConvBlock(3, model, 512)
    ConvBlock(3, model, 512)

    model.add(Flatten())
    FGBlock(model)
    FGBlock(model)
    model.add(Dense(1000, activation='softmax'))
    return model

model = VGG_16()

fpath = get_file('vgg16.h5', FILES_PATH+'vgg16.h5', cache_subdir='models')
model.load_weights(fpath)

batch_size = 4

path = ''

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=batch_size, class_mode = 'categorical'):
    return gen.flow_from_directory(path+dirname, target_size=(224,224), class_model=class_mode, shuffle=shuffle, batch_size=batch_size)


