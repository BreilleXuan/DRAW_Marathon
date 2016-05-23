from DRAW_parameters import *
from DRAW_models import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

sess=tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "weights/drawmodel.ckpt") 


img = cv2.imread('data/test.jpg')
resize_rd = cv2.resize(img, (54,54), interpolation = cv2.cv.CV_INTER_AREA)
imgdone = resize_rd / 255. 
img_flatten = imgdone.reshape(1, 54*54*3)

img_back = img_flatten.reshape(54,54,3)*255.
cv2.imwrite('back.jpg', img_back)

feed_dict = {x:img_flatten}

canvases=sess.run(cs,feed_dict) # generate some examples
canvases=np.array(canvases) # T x batch x img_size


last_img = canvases[-1,0,:]
last_img=1.0/(1.0+np.exp(-last_img)) * 255.
img_out = last_img.reshape(54,54,3) 
print(img_out)
cv2.imwrite('test.jpg', img_out)
