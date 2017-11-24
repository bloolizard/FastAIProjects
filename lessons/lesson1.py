# infinitely flexible function
# all-purpose parameter fitting
# fast and scalable

# neural network
# universal approximation machine

# gradient descent

# cuda
# p2 instance
# t2.micro

# growing use of deep learning at google

# aws-alias.sh

# Python for Data Analysis

# state of the art

# tmux

# ctrl + b split window

# ctrl + b -> % split window

# for Jupyter notebook only
# % matplotlib inline

# unzip *.zip
# unzip -q *.zip (q for quiet)

# datasets
# www.platform.ai/data/

# train valid directories
# make one directory for cat, one for dogs
# 
# ls -l train/dogs | wc -l
#
# ls -l train/cats | wc -l
#
# have a training set and test set
#
#
# test set does not identify
# job: figure out test set
#
# try not to look at test set until finished
#
# * do work on a sample set
#
# directory sample with own train + valid set
#
# 8 in train 
# 4 in valid
#
#

# run on sample or run on anything
# path = 'data/dogscats/'
path = 'data/dogscats/sample'

# a few basic libraries

from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

# conda install
# auto compiles dependencies

# numpy for linear algebra

import utils; reload(utils)
from utils import plots

# use a pretrained VGG model with our Vgg16 class
# 
# imagenet just one thing
# most pictures have more than one thing
#
# downside using pretrained; shortcomings of the data

# imagenet competition winners; make source code and weights available

# visual geometry group (VGG)

# state of the art custom model in 7 lines of code

# as large as you can, but no larger than 64
batch_size = 64

# import our class
from vgg16 import Vgg16
vgg = Vgg16()

# grab a few images at a time for training and validation
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2)

# finetune: use everything you know about pretrained model, and use to detect cats and dogs

vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)

# vgg -> keras -> theano -> cuda
# keras can also sit on top of tensorflow

# change ~/.keras/keras.json to backend to tensorflow
# tensorflow will use all of your gpus
# th
# tf

# ~/.theanorc
# device = gpu
# device = cpu


# batch: how many we look at a time
batches = vgg.get_batches(path + 'train', batch_size=4)
imgs, labels = next(batches)

plots(imgs, titles=labels)

vgg.predict(imgs, True)
vgg.classes[:4]

# use our vgg16 class to finetune a Dogs vs Cats model

batch_size=64
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size)

vgg.finetune(batches)

# if you want to make accuracy higher, rerun cell a bunch of times
vgg.fit(batches, val_batches, nb_epoch=1)





























