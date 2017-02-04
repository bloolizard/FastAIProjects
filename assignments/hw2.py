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



# changes directory to a training set and copies random
path = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/train"
os.chdir(path)
from shutil import copyfile
cat_dir = "/Users/edwizzle/Developer/FastAI/assignments/data/dogs-vs-cats-redux-kernels-edition/sample/train/cats/"
g = glob('cat*.jpg')
shut = np.random.permutation(g)
for i in range(200): copyfile(shut[i], cat_dir + shut[i])