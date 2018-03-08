#! /usr/bin/python

import os.path
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
sess = tf.InteractiveSession()

# define some functions
def weight(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)
    #return tf.Variable(tf.truncated_normal(shape, stddev = .1)) # ValueError: None values not supported

def bias(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

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
W1 = 5; H1 = 5; C1 = 32
W2 = 5; H2 = 5; C2 = 64
C3 = 1024
with tf.name_scope('input_data'):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1,28,28,1])
    tf.summary.image('input', x_image, max_outputs=10)
# 1st conv layer
with tf.name_scope('hidden_conv_1'):
    W_conv1 = weight([W1,H1,1,C1],name='weights')
    b_conv1 = bias([C1],name='bias')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # summaries
    tf.summary.histogram('histogram of W1', W_conv1)
    tf.summary.histogram('histogram of b1', b_conv1)
# 2nd conv layer
with tf.name_scope('hidden_conv_2'):
    W_conv2 = weight([W2,H2,C1,C2],name='weights')
    b_conv2 = bias([C2],name='bias')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # now that the image size has been reduced to 7x7
    # summaries
    tf.summary.histogram('histogram of W2', W_conv2)
    tf.summary.histogram('histogram of b2', b_conv2)
# densely connected layer with 1024 neurons
with tf.name_scope('hidden_fully_connected'):
    W_fc = weight([7*7*C2, C3],name='weights')
    b_fc = bias([C3],name='bias')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*C2])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
    # summaries
    tf.summary.histogram('histogram of W_fc', W_fc)
    tf.summary.histogram('histogram of b_fc', b_fc)
# dropout
with tf.name_scope('dropout'):
    p_keep = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(h_fc, p_keep)
# output layer
with tf.name_scope('output_layer'):
    W_output = weight([C3, 10],name='weights')
    b_output = bias([10],name='bias')
    y = tf.matmul(h_fc_drop, W_output) + b_output
    # summaries
    tf.summary.histogram('histogram of W_out', W_output)
    tf.summary.histogram('histogram of b_out', b_output)
    tf.summary.histogram('histogram of prob_dist_out', y_)

# cost
with tf.name_scope('cost'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
    # summary
    tf.summary.scalar('loss', cross_entropy)

# training op
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)

# accuracy testing
with tf.name_scope('accuracy_test'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # summary
    tf.summary.scalar('accuracy', accuracy)

# summary writer
summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./MNIST_logs/train', sess.graph)
test_writer  = tf.summary.FileWriter('./MNIST_logs/test' , sess.graph)
embedding_writer  = tf.summary.FileWriter('./MNIST_logs/embedding' , sess.graph)

# create metadata for visualizing
test_images = tf.Variable(mnist.test.images, name='test_images')
test_labels = tf.argmax(mnist.test.labels,1).eval(session=sess)
#test_images, test_labels = mnist.test.next_batch(1000)
#test_images = tf.Variable(test_images,name='test_images1')
#test_labels = tf.argmax(test_labels,1).eval(session=sess)
with open('./MNIST_logs/embedding/metadata.tsv', 'w') as metadata_file:
    for label in test_labels:
        metadata_file.write('%d\n' % label)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# training
max_iter = 20000
batch_size = 50
keep_prob = 0.5
sess.run(tf.global_variables_initializer())
for iter in range(max_iter):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    if iter%100 == 0:
        # report accuracy
        train_accuracy = accuracy.eval({x:batch_x, y_:batch_y, p_keep: 1.0}) #  turn off dropout during testing
        test_accuracy = 0.0
        for i in range(100):
            test_x, test_y = mnist.test.next_batch(100)
            test_acc = accuracy.eval({x:test_x, y_:test_y, p_keep:1.0})
            test_accuracy = test_accuracy + test_acc
        test_accuracy = test_accuracy/100.0
        print("iter step %d batch accuracy %f test accuracy %f"%(iter, train_accuracy, test_accuracy))
        # write summary file
        summary_train = sess.run(summary, {x:batch_x, y_:batch_y, p_keep: 1.0})
        summary_test  = sess.run(summary, {x:test_x, y_:test_y, p_keep: 1.0})
        train_writer.add_summary(summary_train, iter)
        test_writer.add_summary(summary_test  , iter)
        train_writer.flush()
        test_writer.flush()
    if iter%1000 == 0:
        # save checkpoint file
        checkpoint_file = os.path.join('./MNIST_logs', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=iter)
        # associate metadata with the embedding.
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add() # You can add multiple embeddings. Here we add only one
        embedding.tensor_name = test_images.name
        embedding.metadata_path = os.path.join('./MNIST_logs/embedding', 'metadata.tsv') # Link this tensor to its metadata file
        projector.visualize_embeddings(embedding_writer, config) # Saves a configuration file that TensorBoard will read during startup
    # train on setp
    train_step.run({x:batch_x, y_:batch_y, p_keep: keep_prob})
train_writer.close()
test_writer.close()
embedding_writer.close()
