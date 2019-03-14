# LearningTensorflow  

## These demos are all based on Tensorflow's low level APIs.
## Tensorflow has witnessed huge changes. For example, Keras has been merged to Tensorflow as high level APIs. In Tensorflow 2.0, dynamic dynamic computational graph replaces static graph as the default pattern. Therefore, some of the demos in this repo may be out-of-date. Just use them as a reference.  

***
New Updates:
* March 8, 2018: adding new demo for reading data via queue-based input pipeline; [source code](https://github.com/SaoYan/LearningTensorflow/blob/master/exp11_user_dataset_low_API_1.py)
* March 27, 2018: adding new demo for reading data using high level API tf.data; [source code](https://github.com/SaoYan/LearningTensorflow/blob/master/exp13_user_dataset_high_API_1.py)

***

Tutorials (in Chinese) on my WeChat Public Account:  
![public-account.jpg](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/public-account.jpg)

## exp1_basic_usage.py:     
This demo helps to understand basic concepts including graph, Session, placeholder, constant and Variable.

## exp2_simple_linear_model.py    
This demo shows how to build a linear regression model.     
The key is to help understand the framework of a model training pipeline in Tensorflow.      
1. Define tensor objects.    
optimizable model parameters: defined as Variable    
fixed model parameters: defined as constant    
model inputs: defined as placeholder (to be "fed" by training or test data)    
2. Define optimization object    
This can be MSE (e.g. linear regression), cross entropy loss (e.g. logistic regression, softmax regression), or other possible ones.    
3. Pick an optimizer    
This can be SGD, SGD with momentum, Adam, and so on. This demo is so simple that SGD is enough.    
4. Run optimization    
In step (1)~(3), we actually build a Computational Graph, which is no more than a model before we RUN it with Session    
In the final step, use Session to run the optimizer (the one defined in step 3).    
Remember to FEED each placeholder when running the graph!

## exp3_SoftmaxRegression_mnist.py    
Let's try something more interesting!    
This demo shows how to build a sofmax regression model for the MNIST dataset.    
* no hiden layer  
* cost function: cross entropy loss    
* momentum optimizer, learning rate 0.1, momentum 0.9    
* batch size 100    
* maximum iteration 1000    
* no regularization, no dropout

## exp4_CNN_mnist.py    
Of course the CNN model cannot be missed!    
This demo shows how to build a CNN model for MNIST dataset.    
* hidden layer: 2 convolution layers, each of which followed by one max-pooling layer    
* output: one fully connected layer (with dropout)    
* cost function: cross entropy loss    
* Adam optimizer, learning rate 1e-4    
* batch size 100    
* maximum iteration 20000    
* using dropout, dropout probability 0.5 (remember using dropout just in training process, no dropout when testing!)

## exp5_Iris_data_set.py
This is a demo for use of Tensorflow high level API tf.contrib.    
For simplicity, use the IRIS dataset and a simple 3-layer feed forward network.

## exp6_Customer_InputFun.py
It is often the case that we need to pre-processing the dataset. Customizing our own input_function is a good choice. Then you can provide the function handle when running the graph.     
For more detail, see [Tensorflow Document](https://www.tensorflow.org/get_started/input_fn).

## exp7~exp9
These three demos are all about Tensorboard. All of them are extended based on exp4. (The model is exactly the same, I just add some extra codes for visualizing on Tensorboard.)

First, you can try Tensorboard and see how amazing it is.       
1. Just run exp9, you will see a folder named 'MNIST_logs' in your current path.    
2. Open command line. Run command
```
tensorboard --logdir MNIST_logs
```    
3. You will see something similar to the following content:    
> TensorBoard 0.4.0 at http://sao:6006 (Press CTRL+C to quit)

4. Open your browser, go to the address shown on your command line

Now you may want to see the details of the demo. Here is the overview:    
Generally speaking, I wrap four types of information for visualization on Tensorboard    
1. Training information. For example, record loss value to plot a trainging curve, or record weight values to plot a histogram for visualizing their distribution.  
![fig1.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/fig1.png)
![fig2.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/fig2.png)
2. Images. In this demo only original MNIST data is visualized. In practice, it may be more useful for you to visualize features of hidden layers.
![fig3.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/fig3.png)
3. Embeddings.  
![fig4.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/fig4.png)
4. Comnputational Graph. In addition to the architecture of the graph, information on running status is also shown (including memory useage, computing time, etc.)  
![fig5.png](https://github.com/SaoYan/LearningTensorflow/blob/master/Figures/fig5.png)

**exp7_TensorBoard**: visualizing training information, images, and simplest version of computational graph    
**exp8_Embedding_Visualization**: add Embeddings for visualization    
**exp9_Graph_Visualization**: add running status information
