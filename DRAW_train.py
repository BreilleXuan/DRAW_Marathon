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
print("Learning Rate: ", learning_rate)
print("QSampler size: ", z_size)
print("Batch Size   : ", batch_size)
print("Encoder Size : ", enc_size)
print("Decoder Size : ", dec_size)

print("--------------------------")

imglist = load_name_list(img_name_file)

for i in range(train_iters):

    namelist = minibatches(imglist, batch_size, shuffle=True)

    for j in range(len(namelist)):
        
        batch_name_list = namelist[j]
        xtrain = loadimg(srcdir, batch_name_list)
        feed_dict={x:xtrain}
        results=sess.run(fetches,feed_dict)
        Lxs[i],Lzs[i],_=results

        if (print_interval + 1) % print_interval == 0:
            print("epoch=%d,iter=%d : Lx: %f Lz: %f" % (i,j,Lxs[i],Lzs[i]))
    
    ckpt_file=os.path.join(FLAGS.data_dir,prefix+str(i+1)+".ckpt")
    print("Model saved in file: %s" % saver.save(sess,ckpt_file))
    

## TRAINING FINISHED ## 

# canvases=sess.run(cs,feed_dict) # generate some examples
# canvases=np.array(canvases) # T x batch x img_size

# out_file=os.path.join(FLAGS.data_dir,"draw_data.npy")
# np.save(out_file,[canvases,Lxs,Lzs])
# print("Outputs saved in file: %s" % out_file)

sess.close()
