import tensorflow as tf

output = tf.nn.l2_normalize([301.42237830162,177.69162777066,326.05302262306,122.9596464932,0.99799561500549,0.99766051769257], dim = 0)
with tf.Session() as sess:
    print(sess.run(output))
