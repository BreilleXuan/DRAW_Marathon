import tensorflow as tf
import numpy as np
import os

from DRAW_load_batch import *
from DRAW_models import *
from DRAW_parameters import *

sess=tf.Session()

print("Model Building Complete...")

print("--------------------------")
print("Start training...")
print("--------------------------")



imglist = load_name_list(img_name_file)

for i in range(train_iters):

    namelist = minibatches(imglist, batch_size, shuffle=True)

    for j in range(len(namelist)):
        name_batch_list = namelist[j]
        xtrain = loadimg(srcdir, name_batch_list)
