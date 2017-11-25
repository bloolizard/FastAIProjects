# david perkins - making learning whole
#
#

# install correct keras version
# pip install keras==1.2

# fine tuning is a process to take a network model
# that has already been trained for a given task
# and make it perform a similar task

path = "data/redux/"

import utils; reload(utils);
from utils import *

from vgg16 import vgg16
vgg = Vgg16()

batch_size=64

batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2)
vgg.finetune(batches)

vgg.fit(batches, val_batches, nb_epoch=1)

vgg.model.save_weights(path + 'results/ft1.h5')

# kg download
# pip install kaggle-cli

# create test and train
#

# todo:

# 1. create validation set and sample
# 2. move to separate dirs for each set
# 3. finetune and train
# 4. submit

# % cd data/redux
# ! (any bash command)
# % cd train
# % mkdir ../valid

g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000):
	os.rename(shuf[i], '../valid/' + shuf[i])

# mkdir ../sample
# mkdir ../sample/train
# mkdir ../sample/valid

from shutil import copyfile
g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(200):
	copyfile(shuf[i], '../sample/train/' + shuf[i])

# % cd ../valid
g = glob('*.jpg') 
for i in range(50):
	copyfile(shuf[i], '../sample/valid/' + shuf[i])

# move to separate dirs for each 

# % cd ../train
# % cd ../valid
# % cd ../sample/train
# % cd ../valid
# % mkdir cats
# % mkdir dogs
# % mv cat.*.jpg cats/
# % mv dog.*.jpg dogs/

from vgg16 import Vgg16
vgg = Vgg16()
batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
vgg.model.save_weights(path + 'results/ft1.h5')

# submit

batches, preds = vgg.test(path + 'test', batch_size = batch_size * 2)
filenames = batches.filenames

preds[:5]
filenames[:5]

save_array(path + 'results/test_preds.dat', preds)
save_array(path + 'results/filenames.dat'. filenames)

preds = load_array('results/test_preds.dat')
filenames = load_array('results/filenames.dat')


# keras has feature called predict_generator
# self.model.predict_generator(test_batches, test_batches.nb_sample)

# test instead of get_batches
# class_model=None
# -> give probabilities instead of labels

from PIL import Image
Image.open('test/' + filenames[0])
isdog = preds[:, 1]
isdog[:5]

ids = [int(f[8:f.find('.')]) for f in filenames]
ids[:5]

subm = np.stack([ids, isdog], axis=1)
subm[:5]

np.savetxt('data/redux/subm98.csv', subm, fmt='%d, %.5f', header='id,label', comments='')

# Jupyter Notebook Only
from IPython.display import FileLink

# on Kaggle
# Question Mark Score ?

# categorical_entropy
# model.compile()

# can't take logs of 1 and 0s
isdog = np.clip(preds[:,1], 0.05, 0.95)

# maybe can play around with clip
isdog = np.clip(preds[:,1], 0.025, 0.975)

# increase your position
vgg.fit(batches, val_batches, nb_epoch=1)

# lr stands for learning rate
vgg.model.optimizer.lr = 0.01
vgg.model.save_weights(path + 'results/ft1.h5')

# maybe change weights filename if you want to have a history and go back

# as well as looking at the overall metrics, it's also a good idea to look at examples of each of:
# 1. a few correct labels at random
# 2. a few incorrect labels at random
# 3. the most correct labels of each class (ie those with highest probability that are correct)
# 4. the most incorrect labels of each class (ie those with highest probability that are incorrect)
# 5. the most uncertain labels (ie those with probability closest to 0.5)

vgg.model.load_weights(path + 'results/ft1.h5')
val_batches, probs = vgg.test(path + 'valid', batch_size = batch_size)

labels = val_batches.classes
filenames = val_batches.filenames

probs = probs[:, 0]
preds = np.round(1-probs)
probs[:8]

preds[:8]

n_view = 4

def plots_idx(idx, titles=None):
	plots([image.load_img(path + 'valid/' + filenames[i]) for i in idx], titles=titles)

#1. A few correct labels at random
correct = np.where(preds==labels)[0]
idx = permutation(correct)[:n_view]
plots_idx(idx, probs[idx])

#2. A few incorrect labels at random
incorrect = np.where(preds != labels)[0]
idx = permutation(incorrect)[:n_view]
plots_idx(idx, probs[idx])

# as above, but dogs
correct_dogs = np.where((preds == 1) & (preds == labels))[0]
most_correct_dogs = np.argsort(probs[correct_digs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], probs[correct_dogs][most_correct_dogs])

#ft stands for fine tune ft.h5

#3. The images we were most confident were cats, but are actually dogs
incorrect_cats = np.where((preds == 0) & (preds != labels))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats])[::-1][:n_view]
plots_idx(incorrect_cats[most_incorrect_cats], probs[incorrect_cats][most_incorrect_cats])

#3. The images we were most confident were dogs, but are actually cats
incorrect_dogs = np.where((preds == 1) & (preds != labels))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[:n_view]
plots_idx(incorrect_dogs[most_incorrect_dogs], probs[incorrect_dogs][most_incorrect_dogs])


#5. The most uncertain labels( ie those with probability closest to 0.5)
most_uncertain = np.argsort(nb.abs(probs-0.5))
plots_idx(most_uncertain[:n_view], probs[most_uncertain])

# 1. create validation set and sample
# 2. move to separate dirs for each set
# 3. finetune and train
# 4. submit

# redo on different dataset

path = "data/state/"
import utils; reload(utils)
from utils import * 

batch_size=64

from vgg16 import Vgg16
vgg = Vgg16()

batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2)

vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=0.01)

# national lung screening data
# took imagenet
# finetune imagenet
#

# finetuning
# visualizing and undrstanding convolutinal networks
#

# matt zeiler
# what networks learn


# gabor filters
# second layer
# layer three -> finding pieces of text, -> edges of natural things
# layer 4 -> certain kinds of dog face
# layer 5 -> eyes
# VGG is 16 layers
# let's keep all of the learned filters, use them, combine to learn cats and dogs rather than imagenet
# starting with a pretrained network, usually always a good idea

# what exactly is fine tuning?
# what exactly is a neural network? 

# which layer should you fine tune from?
# try a few ...

# ctrl + shift + enter to matrix multiply in excel

# xavier intialization

# keras handles weight initialization for you


# linear model from scratch

import math, sys, os, numpy as np
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
from __future__ iport print_function, division
from ipywidgets import interact, interaction, fixed
from ipywidgets.widgets import *
rc('animation', html='html5')
rcParams['figure.figsize'] = 3, 3

% precision 4

np.set_printoptions(precision=4, linewidth=100)

def lin(a, b, x): return a * x + b

a = 3.
b = 8.
n = 30
x = random(n)
y = lin(a, b, x)
x
y

plt.scattter(x,y)

def sse(y, y_pred): return ((y-y_pred)**2).sum()

def loss(y, a, b, x): return sse(y, lin(a,b,x))

def avg_loss(y, a, b, x): return np.sqrt(loss(y, a, b, x)/n)

a_guess = -1.
b_guess = 1.
avg_loss(y, a_guess, b_guess, x)

lr = 0.01

# calculate derivative in wolfram in alpha

def upd():
	global a_guess, b_guess
	y_pred = lin(a_guess, b_guess, x)
	dydb = 2 * (y_pred - y)
	dyda = x* dydb
	a_guess -= lr*dyda.mean()
	b_guess -= lr*dydb.mean()

fig = plt.figure(dpi=100, figsize=(5,4))
plt.scatter(x,y)
line, = plt.plot(x,lin(a_guess, b_guess, x))
plt.close()

def animate(i):
	line.set_ydata(lin(a_guess, b_guess, x))
	for i in range(10): upd()
	return line,

ani = animation.FuncAnimation(fig, animate, np.arange(0, 40), interval=100)
ani

# stochastic gradient descent

# mathematical operation

# local minima

# deep learning, no local minimum

# modern neural network libraries have symbolic derivation, auto calculate derivatives


# do the same thing in Keras

% matplotlib inline
import utils; reload(utils)
from utils import *

x = random((30,2))
y = np.dot(x, [2., 3.]) + 1.
x[:5]

# we can create a simple linear model (Dense() - with no activation - in Keras) and optimize it using SGD
# minimize mean squared error(mse)
# Dense aka known as Fully Connected

lm = Sequential([Dense(1, input_shape=(2,))])
lm.compile(optimizer=SGD(lr=0.1), loss='mse')

lm.evaluate(x, y, verbose=0)

# to do solving
lm.fit(x, y, nb_epoch=5, batch_size=1)

# to get weights
lm.get_weights()

# current imagenet returns 1000 probabilities...

path = 'data/dogscats/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)


batch_size = 100

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model

# Out overall approach here will be:

# 1. get the true labels for every image
# 2. get the 1000 imagenet category predictions for every image
# 3. feed these predictions as inputs to a simple linear model

# Let's by grabbing training and validation batches

val_batches = get_batches(path + 'valid', shuffle=False, batch_size=1)
batches = get_batches(path + 'train', shuffle=False, batch_size=1)

# save numpy arrays very quickly and very little space
import bcolz

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()

def load_array(fname): return bcolz.open(fname)[:]

# ?? get_data


val_data = get_data(val_batches)
trn_data = get_data(batches)

trn_data.shape

# keras returns classes as a single column, so we convert to one hot encoding

def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)

trn_labels.shape


# ... and their 1,000 imagenet probabilies from VCG16 -- these will be the features for our linear model:
trn_features = model.predict(trn_data, batch_size=batch_size)
val_features = model.predict(val_data, batch_size=batch_size)

trn_features.shape

save_array(model_path + 'train_lastlayer_features.bc', trn_features)
save_array(model_path + 'valid_lastlayer_features.bc', val_features)

trn_features = load_array(model_path + 'train_lastlayer_features.bc')
val_features = load_array(model_path + 'valid_lastlayer_features.bc')

trn_features[0]

lm = Sequential([Dense(2, activation='softmax', input_shape=(1000,))])
lm.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics['accuracy'])

batch_size = 64
batch_size = 4

lm.fit(trn_features, trn_labels, nb_epoch=3, batch_size=batch_size, validation_data=(val_features, val_labels))

lm.summary()

# how to simplify matrix multiply
# deep learning
# activation
# -> put linear function through non-linearity

# max(0,x) -> relu

# activation functions
# tanh
# sigmoid
# max(0, x) relu

lm = Sequential([Dense(2, activation='softmax', input_shape-(1000,))])

model.add(Dense(4096, activation='relu'))

# Retrain last layer's linear model

model.pop()
for layer in model.layers: layer.trainable=False

model.add(Dense(2, activation='softmax'))


gen = image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)

def fit_model(model, batches, val_batches, nb_epoch=1):
	model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=nb_epoch, 
		validation_data=val_batches, nb_val_samples=val_batches.N)

opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

fit_model(model, batches, val_batches, nb_epoch=2)

model.save_weights(model_path + 'finetune1.h5')
model.load_weights(model_path + 'finetune1.h5')
model.evaluate(val_data, val_labels)





































































