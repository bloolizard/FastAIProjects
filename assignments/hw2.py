%matplotlib inline

from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import utils; reload(utils);
from utils import plots

import vgg16; reload(vgg16)
from vgg16 import Vgg16

batch_size = 64

# set the path
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/"

sys.path.extend(['/Users/edwizzle/Developer/FastAI/assignments'])

# changes directory to a training set and copies random
# cats
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/train"
os.chdir(path)
from shutil import copyfile
cat_dir = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/train/cats/"
g = glob('cat*.jpg')
shut = np.random.permutation(g)
for i in range(200): copyfile(shut[i], cat_dir + shut[i])


# changes directory to a training set and copies random
# dogs
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/train"
os.chdir(path)
from shutil import copyfile
dog_dir = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/train/dogs/"
g = glob('dog*.jpg')
shut = np.random.permutation(g)
for i in range(200): copyfile(shut[i], dog_dir + shut[i])

# do the same for validation sets
# cat validation
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/train"
os.chdir(path)
from shutil import copyfile
cat_dir = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/valid/cats/"
g = glob('cat*.jpg')
shut = np.random.permutation(g)
for i in range(20): copyfile(shut[i], cat_dir + shut[i])

# dog validation
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/train"
os.chdir(path)
from shutil import copyfile
dog_dir = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/valid/dogs/"
g = glob('dog*.jpg')
shut = np.random.permutation(g)
for i in range(20): copyfile(shut[i], dog_dir + shut[i])


# actual training
root_path = "/Users/edwizzle/Developer/FastAI/assignments/"
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/"

os.chdir(root_path)
batch_size = 64
vgg = Vgg16()

batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)

# save the weights
vgg.model.save_weights('ft1_0123.h5')

test_path = "/home/ubuntu/nbs/data/dogs-vs-cats-redux-kernels-edition/test1"
batches, preds = vgg.test(test_path, batch_size = batch_size * 2)


batches.filenames

utils.save_array(path + 'results/test_preds2.dat', preds)

utils.save_array(path + 'results/filenames2.dat', filenames)

