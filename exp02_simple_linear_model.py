#! /usr/bin/python
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("train a simple linear model by yourself !")

# model parameters
W = tf.Variable(0.3,  tf.float32)
b = tf.Variable(-0.3, tf.float32)

# model inputs & outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# the model
out = W * x + b

# loss function
loss = tf.reduce_sum(tf.square(out - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# training data
x_train = np.random.random_sample((100,)).astype(np.float32)
y_train = np.random.random_sample((100,)).astype(np.float32)

# training
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x:x_train, y:y_train})
        current_loss = sess.run(loss, {x:x_train, y:y_train})
        print("iter step %d training loss %f" % (i, current_loss))
    print(sess.run(W))
    print(sess.run(b))
