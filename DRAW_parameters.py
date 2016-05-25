import tensorflow as tf

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn", True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 

A,B = 54,54 # image width,height
img_size = B*A*3 # the canvas size

img_mean = 133.990371035
img_var = 4586.26211325
img_std = 67.721947057434789

enc_size = 800 # number of hidden units / output size in LSTM
dec_size = 800

read_n = 20 # read glimpse grid width/height
write_n = 20 # write glimpse grid width/height
read_size = 2*read_n*read_n*3 if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n*3 if FLAGS.write_attn else img_size
z_size=100 # QSampler output size
T=12 # number of generation step

img_name_file = "data/namefile.csv" # image name file
srcdir = "data/npy_images/" # image directory

batch_size=10 # training minibatch size
train_iters=100000
learning_rate=3e-4 # learning rate for optimizer
eps=1e-10 # epsilon for numerical stabilitys
ld = 1.

img_jitter = True
# img_jitter = False
jitter = 0.1

print_interval = 1

prefix = "weights/drawmodel_NO" #model save prefix
save_interval = 1

# continue_training = True
continue_training = False
load_model_name = 'weights/drawmodel_NO20.ckpt'
