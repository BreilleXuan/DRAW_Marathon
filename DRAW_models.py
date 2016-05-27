import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell

from DRAW_parameters import *

DO_SHARE=None # workaround for variable_scope(reuse=True)

x = tf.placeholder(tf.float32,shape=(batch_size,img_size)) # input (batch_size * img_size)
e=tf.random_normal((batch_size,z_size), mean=0, stddev=1) # Qsampler noise
lstm_enc = LSTMCell(enc_size, read_size+dec_size) # encoder Op
lstm_dec = LSTMCell(dec_size, z_size) # decoder Op

def linear(x,output_dim):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    w=tf.get_variable("w", [x.get_shape()[1], output_dim]) 
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y) / (2*sigma2))) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy

def attn_window(scope,h_dec,N):
    with tf.variable_scope(scope,reuse=DO_SHARE):
        params=linear(h_dec,5)
    gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
    gx=(A+1)/2*(gx_+1)
    gy=(B+1)/2*(gy_+1)
    sigma2=tf.exp(log_sigma2)
    delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
    return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)

## READ ## 
def read_no_attn(x,x_hat,h_dec_prev):
    return tf.concat(1,[x,x_hat])

def read_attn(x,x_hat,h_dec_prev):
    Fx0,Fy0,gamma0=attn_window("read0",h_dec_prev,read_n)
    Fx1,Fy1,gamma1=attn_window("read1",h_dec_prev,read_n)
    Fx2,Fy2,gamma2=attn_window("read2",h_dec_prev,read_n)
    print(Fx0 == Fx1)
    print(Fy0 == Fy1)
    def filter_img(img,Fx,Fy,gamma,N):
        Fxt=tf.transpose(Fx,perm=[0,2,1])
        img=tf.reshape(img,[-1,B,A])
        glimpse=tf.batch_matmul(Fy,tf.batch_matmul(img,Fxt))
        glimpse=tf.reshape(glimpse,[-1,N*N])
        return glimpse*tf.reshape(gamma,[-1,1])

    x0=filter_img(x[:, :B*A],Fx0,Fy0,gamma0,read_n) # batch x (read_n*read_n*3)
    x1=filter_img(x[:, B*A:2*B*A],Fx1,Fy1,gamma1,read_n) # batch x (read_n*read_n*3)
    x2=filter_img(x[:, 2*B*A:3*B*A],Fx2,Fy2,gamma2,read_n) # batch x (read_n*read_n*3)
    x = tf.concat(1, [x0,x1])
    x = tf.concat(1, [x,x2])

    x_hat0=filter_img(x_hat[:, :B*A],Fx0,Fy0,gamma0,read_n)
    x_hat1=filter_img(x_hat[:, B*A:2*B*A],Fx1,Fy1,gamma1,read_n)
    x_hat2=filter_img(x_hat[:, 2*B*A:3*B*A],Fx2,Fy2,gamma2,read_n)
    x_hat = tf.concat(1, [x_hat0,x_hat1])
    x_hat = tf.concat(1, [x_hat,x_hat2])
    return tf.concat(1,[x,x_hat]) # concat along feature axis

# def read_attn(x,x_hat,h_dec_prev):

#     Fx0,Fy0,gamma0=attn_window("read0",h_dec_prev,read_n)
#     Fx1,Fy1,gamma1=attn_window("read1",h_dec_prev,read_n)
#     Fx2,Fy2,gamma2=attn_window("read2",h_dec_prev,read_n)

#     def filter_img(img,Fx0_,Fy0_,Fx1_,Fy1_,Fx2_,Fy2_,gamma0_,gamma1_,gamma2_,N):

#         Fx0_t=tf.transpose(Fx0_,perm=[0,2,1]) # batch*A*N
#         Fx1_t=tf.transpose(Fx1_,perm=[0,2,1]) # batch*A*N
#         Fx2_t=tf.transpose(Fx2_,perm=[0,2,1]) # batch*A*N

#         img=tf.reshape(img,[-1,B,A,3])    # batch*B*A*3
#                                           # Fy: batch*N*B
#         fydotx_0 = tf.batch_matmul(Fy0_, img[:,:,:,0])
#         fydotx_1 = tf.batch_matmul(Fy1_, img[:,:,:,1])
#         fydotx_2 = tf.batch_matmul(Fy2_, img[:,:,:,2])

#         imgread_0 = tf.batch_matmul(fydotx_0, Fx0_t)
#         imgread_1 = tf.batch_matmul(fydotx_1, Fx1_t)
#         imgread_2 = tf.batch_matmul(fydotx_2, Fx2_t)

#         glimpse_0 = tf.reshape(imgread_0, [-1, N*N]) * tf.reshape(gamma0_,[-1,1])
#         glimpse_1 = tf.reshape(imgread_1, [-1, N*N]) * tf.reshape(gamma1_,[-1,1])
#         glimpse_2 = tf.reshape(imgread_2, [-1, N*N]) * tf.reshape(gamma2_,[-1,1])

#         glimpse = tf.concat(1, [glimpse_0, glimpse_1])
#         glimpse = tf.concat(1, [glimpse, glimpse_2])
    
#         return glimpse

#     x    =filter_img(x,    Fx0,Fy0,Fx1,Fy1,Fx2,Fy2,gamma0,gamma1,gamma2,read_n) # batch x (read_n*read_n*3)
#     x_hat=filter_img(x_hat,Fx0,Fy0,Fx1,Fy1,Fx2,Fy2,gamma0,gamma1,gamma2,read_n)

#     return tf.concat(1,[x,x_hat]) # concat along feature axis

read = read_attn if FLAGS.read_attn else read_no_attn

## ENCODE ## 
def encode(state,input):
    """
    run LSTM
    state = previous encoder state
    input = cat(read,h_dec_prev)
    returns: (output, new_state)
    """
    with tf.variable_scope("encoder",reuse=DO_SHARE):
        return lstm_enc(state, input)

## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##

def sampleQ(h_enc):
    """
    Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
    mu is (batch,z_size)
    """
    with tf.variable_scope("mu",reuse=DO_SHARE):
        mu=linear(h_enc,z_size)
    with tf.variable_scope("sigma",reuse=DO_SHARE):
        logsigma=linear(h_enc,z_size)
        sigma=tf.exp(logsigma)
    return (mu + sigma*e, mu, logsigma, sigma)

## DECODER ## 
def decode(state,input):
    with tf.variable_scope("decoder",reuse=DO_SHARE):
        return lstm_dec(state, input)

## WRITER ## 
def write_no_attn(h_dec):
    with tf.variable_scope("write",reuse=DO_SHARE):
        return linear(h_dec,img_size)

# def write_attn(h_dec):
#     with tf.variable_scope("writeW",reuse=DO_SHARE):
#         w=linear(h_dec,write_size) # batch x (write_n*write_n*3)
#     N=write_n
#     w=tf.reshape(w, [batch_size,N,N,3])
#     Fx,Fy,gamma=attn_window("write", h_dec,write_n)
#     Fyt=tf.transpose(Fy, perm=[0,2,1])
#     # wr=tf.batch_matmul(Fyt, tf.batch_matmul(w,Fx))
#     # wr=tf.reshape(wr,[batch_size,B*A])

#     wFx_0 = tf.batch_matmul(w[:,:,:,0],Fx)
#     wFx_1 = tf.batch_matmul(w[:,:,:,1],Fx)
#     wFx_2 = tf.batch_matmul(w[:,:,:,2],Fx)

#     wr_0 = tf.batch_matmul(Fyt, wFx_0)
#     wr_1 = tf.batch_matmul(Fyt, wFx_1)
#     wr_2 = tf.batch_matmul(Fyt, wFx_2)

#     wr_0 = tf.reshape(wr_0, [batch_size, B*A])
#     wr_1 = tf.reshape(wr_1, [batch_size, B*A])
#     wr_2 = tf.reshape(wr_2, [batch_size, B*A])

#     wr = tf.concat(1, [wr_0, wr_1])
#     wr = tf.concat(1, [wr, wr_2])

#     return wr*tf.reshape(1.0/gamma,[-1,1])

def write_attn(h_dec):
    with tf.variable_scope("writeW",reuse=DO_SHARE):
        w=linear(h_dec,write_size) # batch x (write_n*write_n*3)
        
    N=write_n
    w=tf.reshape(w, [batch_size,N,N,3])

    Fx0,Fy0,gamma0=attn_window("write0", h_dec,write_n)
    Fx1,Fy1,gamma1=attn_window("write1", h_dec,write_n)
    Fx2,Fy2,gamma2=attn_window("write2", h_dec,write_n)

    Fy0t=tf.transpose(Fy0, perm=[0,2,1])
    Fy1t=tf.transpose(Fy1, perm=[0,2,1])
    Fy2t=tf.transpose(Fy2, perm=[0,2,1])
    # wr=tf.batch_matmul(Fyt, tf.batch_matmul(w,Fx))
    # wr=tf.reshape(wr,[batch_size,B*A])

    wFx_0 = tf.batch_matmul(w[:,:,:,0],Fx0)
    wFx_1 = tf.batch_matmul(w[:,:,:,1],Fx1)
    wFx_2 = tf.batch_matmul(w[:,:,:,2],Fx2)

    wr_0 = tf.batch_matmul(Fy0t, wFx_0)
    wr_1 = tf.batch_matmul(Fy1t, wFx_1)
    wr_2 = tf.batch_matmul(Fy2t, wFx_2)

    wr_0 = tf.reshape(wr_0, [batch_size, B*A])*tf.reshape(1.0/gamma0,[-1,1])
    wr_1 = tf.reshape(wr_1, [batch_size, B*A])*tf.reshape(1.0/gamma1,[-1,1])
    wr_2 = tf.reshape(wr_2, [batch_size, B*A])*tf.reshape(1.0/gamma2,[-1,1])

    wr = tf.concat(1, [wr_0, wr_1])
    wr = tf.concat(1, [wr, wr_2])

    return wr

write=write_attn if FLAGS.write_attn else write_no_attn

## STATE VARIABLES ## 

cs=[0]*T # sequence of canvases
mus,logsigmas,sigmas=[0]*T,[0]*T,[0]*T # gaussian params generated by SampleQ. We will need these for computing loss.
# initial states
h_dec_prev=tf.zeros((batch_size,dec_size))
enc_state=lstm_enc.zero_state(batch_size, tf.float32)
dec_state=lstm_dec.zero_state(batch_size, tf.float32)

## DRAW MODEL ## 

# construct the unrolled computational graph
for t in range(T):
    c_prev = tf.zeros((batch_size,img_size)) if t==0 else cs[t-1]
    x_hat=x-tf.sigmoid(c_prev) # error image
    r=read(x,x_hat,h_dec_prev)
    h_enc,enc_state=encode(tf.concat(1,[r,h_dec_prev]), enc_state)
    z,mus[t],logsigmas[t],sigmas[t]=sampleQ(h_enc)
    h_dec,dec_state=decode(z, dec_state)
    cs[t]=c_prev+write(h_dec) # store results
    h_dec_prev=h_dec
    DO_SHARE=True # from now on, share variables

## LOSS FUNCTION ## 

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# reconstruction term appears to have been collapsed down to a single scalar value (rather than one per item in minibatch)
x_recons=tf.nn.sigmoid(cs[-1])

# after computing binary cross entropy, sum across features then take the mean of those sums across minibatches
Lx=tf.reduce_sum(binary_crossentropy(x,x_recons),1) # reconstruction term
Lx=tf.reduce_mean(Lx)

kl_terms=[0]*T
for t in range(T):
    mu2=tf.square(mus[t])
    sigma2=tf.square(sigmas[t])
    logsigma=logsigmas[t]
    kl_terms[t]=0.5*tf.reduce_sum(mu2+sigma2-2*logsigma,1)-T*.5 # each kl term is (1xminibatch)
KL=tf.add_n(kl_terms) # this is 1xminibatch, corresponding to summing kl_terms from 1:T
Lz=tf.reduce_mean(KL) # average over minibatches

cost=Lx + ld * Lz


