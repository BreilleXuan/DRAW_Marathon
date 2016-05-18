import tensorflow as tf

batch = 10
B = 10
A = 5
N = 2

img = tf.ones((batch,B,A,3))
fxt = tf.ones((batch,A,N))
fy = tf.ones((batch,N,B))

fydotx_0 = tf.batch_matmul(fy, img[:,:,:,0])
fydotx_1 = tf.batch_matmul(fy, img[:,:,:,1])
fydotx_2 = tf.batch_matmul(fy, img[:,:,:,2])

imgread_0 = tf.batch_matmul(fydotx_0, fxt)
imgread_1 = tf.batch_matmul(fydotx_1, fxt)
imgread_2 = tf.batch_matmul(fydotx_2, fxt)

glimpse_0 = tf.reshape(imgread_0, [-1, N*N])
glimpse_1 = tf.reshape(imgread_1, [-1, N*N])
glimpse_2 = tf.reshape(imgread_2, [-1, N*N])

glimpse = tf.concat(1, [glimpse_0, glimpse_1])
glimpse = tf.concat(1, [glimpse, glimpse_2])

sess = tf.Session()
output = sess.run(glimpse)
print(output)