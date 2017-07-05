#! /usr/bin/python

print("train a simple linear model by yourself !")

import tensorflow as tf
sess = tf.Session()

# model parameters
W = tf.Variable(0.3,  tf.float32)
b = tf.Variable(-0.3, tf.float32)
init = tf.global_variables_initializer()
sess.run(init)

# model inputs & outputs
x = tf.placeholder(tf.float32)
model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training
for i in range(1000):
	sess.run(train, {x:x_train, y:y_train})
	print("iter step %d training loss %f"%(i, sess.run(loss, {x:x_train, y:y_train})))

# evaluate traning accuracy
curr_W, curr_b, curr_loss = sess.run([W,b,loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
