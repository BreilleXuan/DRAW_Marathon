from DRAW_parameters import *
from DRAW_models import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

sess=tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "data/drawmodel_2_5_100_64.ckpt") 

def img_genr(i):

    canvas_matrix=[0]*T
    dec_state_ = lstm_dec.zero_state(batch_size, tf.float32)

    for t in range(T):
        c_prev_ = tf.zeros((batch_size,img_size)) if t==0 else canvas_matrix[t-1]
        z_tensor = tf.truncated_normal((batch_size, z_size))
        h_dec_,dec_state_=decode(z_tensor, dec_state_)
        canvas_matrix[t]=c_prev_ + write(h_dec_) # store results

    canvases=np.array(sess.run(canvas_matrix)) # T * batch_size * img_size
    canvases=1.0/(1.0+np.exp(-canvases))

    final_img = canvases[-1]

    img = final_img[0]

    out_img = img.reshape(B, A)

    plt.matshow(out_img,cmap=plt.cm.gray)
    imgname='genr_mnist_'+str(i+1)+'.jpg' 
    plt.savefig("img/"+imgname)
    plt.close()


    # for t in range(T):
    #     img = canvases[t,0,:]
    #     out_img = img.reshape(B, A)

    #     plt.matshow(out_img,cmap=plt.cm.gray)
    #     imgname='seq_'+str(i) + '_' + str(t+1)+'.png' 
    #     plt.savefig("img/"+imgname)  
    #     plt.close()      

for i in range(10):
    print(i+1)
    img_genr(i)

