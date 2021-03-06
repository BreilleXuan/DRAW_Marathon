import cv2
import numpy as np
from File.file_csv import *
from DRAW_parameters import *

def jitter(p, img):
	l, w = img.shape[0], img.shape[1] # l:height; w:width
	a, b = np.floor(l * p) - 1, np.floor(w * p) - 1
	rad = np.random.rand(2)
	row_start = np.floor(a * rad[0])
	col_start = np.floor(b * rad[1])
	row_end = min(row_start + np.floor((1-p)*l), l) 
	col_end = min(col_start + np.floor((1-p)*w), w)
	outimg = cv2.resize(img[row_start:row_end, col_start:col_end, :],(l, w))
	return outimg

def img_to_npy(namelist):
	for name in namelist:
		print(name)
		current_img = 'data/cutted_images/' + name
		rd = cv2.imread(current_img) # height, width, channel
		np.save('data/npy_images/'+name.split('.')[0]+'.npy', rd)

def loadimg(srcdir, names, w=54, h=54, p=0.1):
	n = len(names)
	imgset = np.zeros((n, w*h*3))
	
	for i in range(n):
		current_img = names[i].split('.')[0]
		current_img = srcdir + current_img + '.npy'
		rd = np.load(current_img)
		resize_rd = cv2.resize(rd, (w,h), interpolation = cv2.cv.CV_INTER_AREA)
		imgdone = resize_rd / 255.
		if img_jitter == True:
			imgdone = jitter(0.1, imgdone)
		img_flatten = imgdone.reshape(1, w*h*3)
		imgset[i] = img_flatten
	return imgset

def minibatches(imglist, batchsize, shuffle=True):
    if shuffle:
        indices = np.arange(len(imglist))
        np.random.shuffle(indices)
    output = []
    for start_idx in range(0, len(imglist) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        output.append(imglist[excerpt]) 
    return output

def load_name_list(img_name_file):
	return np.array(loadcsv(img_name_file)[0])


if __name__ == '__main__':
	namelist = load_name_list("data/namefile.csv")
	img_to_npy(namelist)


