import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("classify the MNIST dataset using CNN ~~")

# define some functions
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# load the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot = True)

# inputs & outputs
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

# build CNN model
W1 = 5; H1 = 5; C1 = 32
W2 = 5; H2 = 5; C2 = 64
C3 = 1024
# 1st conv layer
W_conv1 = weight([W1,H1,1,C1])
b_conv1 = bias([C1])
h_conv1 = conv2d(x, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 2nd conv layer
W_conv2 = weight([W2,H2,C1,C2])
b_conv2 = bias([C2])
h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2) # feature size: 7x7xC2
# densely connected layer with 1024 neurons
W_fc = weight([7,7,C2,C3])
b_fc = bias([C3])
h_fc = conv2d(h_pool2, W_fc, padding='VALID') + b_fc
h_fc = tf.nn.relu(h_fc)
# dropout
p_keep = tf.placeholder(tf.float32)
h_fc_drop = tf.nn.dropout(h_fc, p_keep)
# output layer
W_output = weight([1,1,C3,10])
b_output = bias([10])
y = conv2d(h_fc_drop, W_output, padding='VALID') + b_output
y = tf.squeeze(y, [1,2]) # shape batchx1x1x10---->batchx10

# cost
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
loss = tf.reduce_mean(cross_entropy) # average over the batch
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)

# accuracy testing
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training
max_iter = 20000
batch_size = 64
keep_prob = 0.5
with tf.Session() as sess:
    # initialize all parameters
    sess.run(tf.global_variables_initializer())
    for iter in range(max_iter):
        # extract a batch of training data from MNIST dataset
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((-1,28,28,1))
        # update parameters
        sess.run(train_step, {x:batch_x, y_:batch_y, p_keep: keep_prob})
        if iter%100 == 0:
            # the accuracy over the training batch
            train_accuracy = sess.run(accuracy, {x:batch_x, y_:batch_y, p_keep: 1.0}) #  turn off dropout during testing
            # the accuracy over the whole test set
            # unable to load the whole test set at one time because of memory limitation
            test_accuracy = 0.0
            for i in range(100):
                test_x, test_y = mnist.test.next_batch(100)
                test_x = test_x.reshape((-1,28,28,1))
                test_acc = sess.run(accuracy, {x:test_x, y_:test_y, p_keep:1.0})
                test_accuracy = test_accuracy + test_acc
            test_accuracy = test_accuracy/100.0
            # print the result
            print("iter step %d batch accuracy %f test accuracy %f" %
                     (iter, train_accuracy, test_accuracy))
