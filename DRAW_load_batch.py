import cv2
import numpy as np
from File.file_csv import *

def jitter(p, img):
	l, w = img.shape[0], img.shape[1]
	a, b = np.floor(l * p) - 1, np.floor(w * p) - 1
	rad = np.random.rand(2)
	row_start = int(a * rad[0])
	col_start = int(b * rad[1])
	row_end = row_start + np.floor((1-p)*l)
	col_end = col_start + np.floor((1-p)*w)
	outimg = cv2.resize(img[row_start:row_end, col_start:col_end, :],(l, w))
	return outimg

def loadimg(srcdir, names, w=54, h=54, p=0.1):
	n = len(names)
	imgset = np.zeros((n, w*h*3))
	
	for i in range(n):
		image = names[i]
		img = srcdir + image
		print(img)
		rd = cv2.imread(img)
		print(rd)
		resize_rd = cv2.resize(rd, (w,h), interpolation = cv2.cv.CV_INTER_AREA)
		imgdone = resize_rd / 255.
		img_jitter = jitter(0.1, imgdone)
		img_flatten = img_jitter.reshape(1, w*h*3)
		imgset[i] = img_flatten
	return imgset

def iterate_minibatches(imglist, batchsize, shuffle=True):
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
	srcdir = "data/cutted_images/"
	img_name = iterate_minibatches(namelist, 4, shuffle=True)

	name_batch = img_name[0]
	print(name_batch)

	loadimg(srcdir, name_batch)



