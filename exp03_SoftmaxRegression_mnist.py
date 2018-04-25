import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("classify the MNIST dataset using softmax regression")

# load the dataset
mnist = input_data.read_data_sets("./MNIST_data", one_hot = True) # labels are "one-hot vectors"

# model parameters
W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))
learning_rate = 0.1
momentum = 0.9
batch_size = 64
max_iter = 1000

# model inputs & outputs
x = tf.placeholder(tf.float32, [None, 784]) # "None" means that a dimension can be of any length
y_ = tf.placeholder(tf.float32, [None, 10])

# creat model
y = tf.matmul(x, W) + b

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_step = optimizer.minimize(cross_entropy)

# accuracy test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # this gives us a list of booleans
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # booleans->float32

# training
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for iter in range(3000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        test_x = mnist.test.images
        test_y = mnist.test.labels
        sess.run(train_step, {x:batch_x, y_:batch_y})
        if iter%10 == 0:
            train_loss = sess.run(cross_entropy,{x:batch_x,y_:batch_y})
            train_accuracy = sess.run(accuracy, {x:batch_x,y_:batch_y})
            test_accuracy = sess.run(accuracy, {x: test_x, y_: test_y})
            print("iter step %d, loss %f, training accuracy %f, test accuracy %f" %
                (iter,train_loss,train_accuracy,test_accuracy))
