import cv2
import numpy as np
from File.file_csv import *


namelist = load_name_list("data/namefile.csv")
srcdir = "data/cutted_images/"
img_name = iterate_minibatches(namelist, 4, shuffle=True)

name_batch = img_name[0]
print(name_batch)

loadimg(srcdir, name_batch)
