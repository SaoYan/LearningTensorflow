#! /usr/bin/python

# This is an example code for getting started with tensorflow
print("getting started with tensorflow\n")

# import tensorflow
print("import...\n")
import tensorflow as tf

# build your first computational graph
# with constant nodes
print("\nbuild computational graph with constant nodes...\n")
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 impicitly
node3 = tf.add(node1, node2)
print([node1, node2, node3])

# this is just BUILDING the graph
# to actually evaluate the graph, you have to run with a SESSION
print("\nrun the graph...\n")
sess = tf.Session()
print(sess.run([node1, node2, node3]))

# you may want a more flexible graph
# with nodes which can FEEDED with exteral inputs
# to do this we use PLACEHODER
print("\nbuild and graph with exteral inputs...\n")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder = a + b # opporator + provides a shotcut for tf.add

# to FEED the PLACEHODERs
# use the feed_dict parameter
print("\nrun the graph...\n")
data1 = {a: 3, b: 2.5}
print(sess.run(adder,feed_dict = data1))
data2 = {a: [1, 2.5], b: [5, 4]}
print(sess.run(adder, data2)) # "deed_dict" can be omited

# we can continue to make the graph more complex
print("\na more complex graph...\n")
add_and_triple = adder * 3 # for a shotcut just use "adder*3"
print(sess.run(add_and_triple, {a:1.0, b: 3.5}))

# in machine learning there are various parameters (weights, bias, etc.)
# use VARIABLE
# having parameters makes a model trainable
print("\nbuild a trainable model...\n")
W = tf.Variable(.3, tf.float32)
b = tf.Variable(-.3, tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# you have to explicitly initialize the VARIABLEs
print("\ninitialize model parameters...\n")
init = tf.global_variables_initializer()
sess.run(init) # note that the VARIABLEs are uninitialized until we call sess.run

# since x is a placeholder, we can feed it and evaluate the model
print("\nrun the model...\n")
print(sess.run(linear_model, {x: 1.0}))
print(sess.run(linear_model, {x: [1,2,3,4]})) # we can evaluate model for several values of x simultaneously

# you can change the value of variables
print("\nassign new values to variables...\n")
W_ = tf.assign(W, 1.0)
b_ = tf.assign(b, -1.0)
sess.run([W_, b_]) # do remember to run !
print(sess.run(linear_model, {x: [1,2,3,4]}))

# now you want to know how good your model is
# you need a y placeholder to provide desired labels
# you also need a loss function
print("\nassess how good the model is...\n")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) # square error
loss = tf.reduce_sum(squared_deltas) # sum over whole data set
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# now you can train your model ~~
print("\ndo gradient descent...\n")
optimizer = tf.train.GradientDescentOptimizer(0.01) # learning rate: 0.01
train = optimizer.minimize(loss)
sess.run(init) # reset values to initialization
for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
	#print("iter step %d training loss %f"%(i, sess.run(loss, {x:x_train, y:y_train})))
print(sess.run([W,b]))
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
