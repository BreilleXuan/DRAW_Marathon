from DRAW_load_batch import *
from DRAW_parameters import *


imglist = load_name_list(img_name_file)

for i in range(train_iters):

    namelist = minibatches(imglist, batch_size, shuffle=True)

    for j in range(len(namelist)):
        name_batch_list = namelist[j]
        xtrain = loadimg(srcdir, name_batch_list)
