import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import os

from DRAW_load_batch import *
from DRAW_models import *
from DRAW_parameters import *

optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
# optimizer=tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
grads=optimizer.compute_gradients(cost)
for i,(g,v) in enumerate(grads):
    if g is not None:
        grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
train_op=optimizer.apply_gradients(grads)

## RUN TRAINING ## 

data_directory = os.path.join(FLAGS.data_dir, "mnist")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train # binarized (0-1) mnist data

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

imglist = load_name_list(img_name_file)
for i in range(train_iters):
    namelist = iterate_minibatches(imglist, batch_size, shuffle=True)
    xtrain = loadimg(srcdir, namelist, w=A, h=B, p=jitter)
    # xtrain,_=train_data.next_batch(batch_size) # xtrain is (batch_size x img_size)
    feed_dict={x:xtrain}
    results=sess.run(fetches,feed_dict)
    Lxs[i],Lzs[i],_=results
    if i % print_interval==0:
        print("iter=%d : Lx: %f Lz: %f" % (i,Lxs[i],Lzs[i]))
    
    if (i+1) % save_interval == 0:
        ckpt_file=os.path.join(FLAGS.data_dir,prefix+str(i+1)+".ckpt")
        print("Model saved in file: %s" % saver.save(sess,ckpt_file))
    

## TRAINING FINISHED ## 

canvases=sess.run(cs,feed_dict) # generate some examples
canvases=np.array(canvases) # T x batch x img_size

out_file=os.path.join(FLAGS.data_dir,"draw_data.npy")
np.save(out_file,[canvases,Lxs,Lzs])
print("Outputs saved in file: %s" % out_file)

sess.close()
