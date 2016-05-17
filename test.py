from DRAW_parameters import *
from DRAW_load_batch import *
import numpy as np

imglist = load_name_list(img_name_file)
a = iterate_minibatches(imglist, batch_size, shuffle=True)








