#! /usr/bin/python

print("classify the MNIST dataset using CNN ~~")

import tensorflow as tf
sess = tf.InteractiveSession()

# define some functions
def weight(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
	#return tf.Variable(tf.truncated_normal(shape, stddev = .1)) # ValueError: None values not supported

def bias(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# load the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)

# build CNN model
# initialize weights with small random values for symmetry breaking
# initialize weights with positive values for preventing "dead ReLU neurons"
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W1 = 5; H1 = 5; C1 = 32
W2 = 5; H2 = 5; C2 = 64
C3 = 1024
# 1st conv layer
W_conv1 = weight([W1,H1,1,C1])
b_conv1 = bias([C1])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 2nd conv layer
W_conv2 = weight([W2,H2,C1,C2])
b_conv2 = bias([C2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # now that the image size has been reduced to 7x7
# densely connected layer with 1024 neurons
W_fc = weight([7*7*C2, C3])
b_fc = bias([C3])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*C2])
h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
# dropout
p_keep = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(h_fc, p_keep)
# output layer
W_output = weight([C3, 10])
b_output = bias([10])
y = tf.matmul(h_fc_drop, W_output) + b_output

# cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(cross_entropy)

# accuracy testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
max_iter = 20000
batch_size = 100
keep_prob = 0.5
sess.run(tf.global_variables_initializer())
for iter in range(max_iter):
	batch_x, batch_y = mnist.train.next_batch(batch_size)
	if iter%100 == 0:
		train_accuracy = accuracy.eval({x:batch_x, y_:batch_y, p_keep: 1.0}) #  turn off dropout during testing
		test_accuracy = 0.0
		for i in range(100):
			test_x, test_y = mnist.test.next_batch(100)
			test_acc = accuracy.eval({x:test_x, y_:test_y, p_keep:1.0})
			test_accuracy = test_accuracy + test_acc
		test_accuracy = test_accuracy/100.0
		print("iter step %d batch accuracy %f test accuracy %f"%(iter, train_accuracy, test_accuracy))
	train_step.run({x:batch_x, y_:batch_y, p_keep: keep_prob})

test_accuracy = 0.0
for i in range(100):
	test_x, test_y = mnist.test.next_batch(100)
	test_acc = accuracy.eval({x:test_x, y_:test_y, p_keep:1.0})
	test_accuracy = test_accuracy + test_acc
test_accuracy = test_accuracy/100.0
print ("final test accuracy %f"%test_accuracy)
