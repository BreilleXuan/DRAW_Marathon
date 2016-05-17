import tensorflow as tf
import numpy as np
import os

from DRAW_load_batch import *
from DRAW_models import *
from DRAW_parameters import *

print("Building model...")
optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
# optimizer=tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

fetches=[]
fetches.extend([Lx,Lz,train_op])
Lxs=[0]*train_iters
Lzs=[0]*train_iters

sess=tf.InteractiveSession()

saver = tf.train.Saver() # saves variables learned during training
if continue_training:
    saver.restore(sess, load_model_name) # to restore from model, uncomment this line
else:
    tf.initialize_all_variables().run()

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
