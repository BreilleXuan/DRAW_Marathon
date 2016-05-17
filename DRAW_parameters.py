import tensorflow as tf

tf.flags.DEFINE_string("data_dir", "", "")
tf.flags.DEFINE_boolean("read_attn", True, "enable attention for reader")
tf.flags.DEFINE_boolean("write_attn",True, "enable attention for writer")
FLAGS = tf.flags.FLAGS

## MODEL PARAMETERS ## 

A,B = 54,54 # image width,height
img_size = B*A # the canvas size

enc_size = 800 # number of hidden units / output size in LSTM
dec_size = 800

read_n = 12 # read glimpse grid width/height
write_n = 12 # write glimpse grid width/height
read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
write_size = write_n*write_n if FLAGS.write_attn else img_size
z_size=100 # QSampler output size
T=32 # number of generation step

img_name_file = 'labels/namelist.txt' # image name file
srcdir = "/img" # image directory
jitter = 0.1

batch_size=16 # training minibatch size
train_iters=10000
learning_rate=3e-3 # learning rate for optimizer
eps=1e-10 # epsilon for numerical stability
ld = 1.

print_interval = 1

prefix = "weights/drawmodel_NO" #model save prefix
save_interval = 500

continue_training = True
# continue_training = False
load_model_name = 'weights/drawmodel_2_5_100_64.ckpt'
