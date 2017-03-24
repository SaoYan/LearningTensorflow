# LearningTensorflow        
This is how I get started with Tensorflow!
===
 
Folder 'Get Started' includes:
---

### exp1_basic_usage.py:     
Just trying basic useage of tensorflow.     
This demo helps to understand basic concepts including graph, Session, placeholder, constant and Variable.

### exp2_simple_linear_model.py    
This is my first toy experiment. In this demo I build a linear regression model.     
The key is to help understand the framework of training a model.      
1) Define tensor objects.    
optimizable model parameters: defined as Variable    
fixed model parameters: defined as constant    
model inputs: defined as placeholder (to be FEEDed by training or test datas)    
2) Define optimization object    
This can be MSE (e.g. linear regression, this demo), cross entropy loss (e.g. logistic regression, softmax regression), and othe possible ones.    
3) Pick an optimizer    
This can be gradient descent, momentum, Adam, and so on. This demo is so simple that GradientDescentOptimizer is enough.    
4) Run optimization    
In step (1)~(3), we actually build a Computational Graph, which is no more than a model before we RUN it with Session    
In the final step, use Session to run the optimizer (the one defined in step 3).    
Remember to FEED each placeholder when running the graph!

### exp3_SoftmaxRegression_mnist.py    
Time to try something more interesting!    
In this demo, I build a sofmax regression model for the MNIST dataset.    
1) no hiden layer    
2) cost function: cross entropy loss    
3) momentum optimizer, learning rate 0.1, momentum 0.9    
4) batch size 100    
5) maximum iteration 1000    
6) no regularization, no dropout

### exp4_CNN_mnist.py    
Of course the CNN model cannot be missed!    
In this demo I build a CNN model for MNIST dataset.    
1) hidden layer: 2 convolution layers, each of which followed by one max-pooling layer    
2) output: one fully connected layer (with dropout)    
3) cost function: cross entropy loss    
4) Adam optimizer, learning rate 1e-4    
5) batch size 100    
6) maximum iteration 20000    
7) using dropout, dropout probability 0.5 (remember using dropout just in training process, no dropout when testing!)

### exp5_Iris_data_set.py
This is a demo for use of Tensorflow high level API tf.contrib.    
For simplicity, use the IRIS dataset and a simple 3-layer feed forward network.

### exp6_Customer_InputFun.py
It is often the case that we need to pre-processing the dataset. Customizing our own input_function is a good choice.     
In this demo, I define a simple input_function. We then provide the function handle when run the graph.     
For more detail, see   
https://www.tensorflow.org/get_started/input_fn

### exp7_TensorBoard.py  exp8_Embedding_Visualization.py  exp9_Graph_Visualization.py
These three demos are all about Tensorboard. All of them are a modification of exp4. (The model is exactly the same, I just add some extra codes for visualizing on Tensorboard.)

First, you can try Tensorboard and see how amazing it is.       
1) Just run exp9, you will see a folder named 'MNIST_logs' in your current path.    
2) Open command line. Run command 'tensorboard --logdir=logpath', where 'logpath' is the path of 'MNIST_logs'.    
3) You will see the following content (the specific port number may not be the same):    
Starting TensorBoard 39 on port 6006    
(You can navigate to http://127.0.1.1:6006)    
4) Open your browser, go to the address http://127.0.1.1:6006 (the address you get from command line)

Now you may want to see the details of the demo. Here is the overview:    
Generally speaking, I send four types of information to Tensorboard    
1) Training information. For example, record loss value to plot a trainging curve, or record weight values to plot a histogram for visualizing their distribution.
![fig1.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Get%20Started/see%20me/fig1.png)
![fig2.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Get%20Started/see%20me/fig2.png)
2) Images. We can resize the MNIST data to proper size and visualize them.
![fig3.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Get%20Started/see%20me/fig3.png)
3) Embeddings. For example, we can use tensorboard to visualize the feature space or data space.
![fig4.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Get%20Started/see%20me/fig4.png)
4) Program running information, including memory useage, computing time, etc.
![fig5.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Get%20Started/see%20me/fig5.png)
exp7 only includes training information and images.    
exp8 steps one more step and adds embeddings.    
exp9 adds program running information, this is the most complete version!
