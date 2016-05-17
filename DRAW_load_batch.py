import cv2
import numpy as np

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
		print srcdir + '/' + image
		rd = cv2.imread(srcdir + '/' + image)
		resize_rd = cv2.resize(rd, (w,h), interpolation = cv2.cv.CV_INTER_AREA)
		imgdone = rd / 255.
		img_jitter = jitter(0.1, imgdone)
		img_flatten = img_jitter.reshape(1, w*h*3)
		imgset[i] = img_flatten

	return imgset

def iterate_minibatches(imglist, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(imglist))
        np.random.shuffle(indices)
    for start_idx in range(0, len(imglist) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield imglist[excerpt]

def load_name_list(img_name_file):
	TrainNames = []
	for row in open(img_name_file):
		rd = row.split()
		TrainNames.append(rd[0])
	return TrainNames


if __name__ == '__main__':
	avg = np.load('avg.npy')
	names = ['001496e2-973a-3b59-25ae-4b389c976d8c.jpg', '002775fe-9ddc-341c-75f4-d6be1e012bf0.jpg']
	print loadimg('image_resize', names, avg)
