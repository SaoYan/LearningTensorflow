# LearningTensorflow        
Get started with Tensorflow!
===
 
Folder 'Get Started' includes:
---

### ------exp1_basic_usage.py:     
Just trying basic useage of tensorflow.     
This demo helps to understand basic concepts including graph, Session, placeholder, constant and Variable.

### ------exp2_simple_linear_model.py    
This is my first toy experiment. In this demo I build a linear regression model.     
The key is to help understand the framework of training a model.      
(1) Define tensor objects.    
optimizable model parameters --- defined as Variable    
fixed model parameters       --- defined as constant    
model inputs                 --- defined as placeholder (to be FEEDed by training or test datas)    
(2) Define optimization object    
This can be MSE (e.g. linear regression, this demo), cross entropy loss (e.g. logistic regression, softmax regression), and othe possible ones.    
(3) Pick an optimizer    
This can be gradient descent, momentum, Adam, and so on. This demo is so simple that GradientDescentOptimizer is enough.    
(4) Run optimization    
In step (1)~(3), we actually build a Computational Graph, which is no more than a model before we RUN it with Session    
In the final step, use Session to run the optimizer (the one defined in step 3).    
Remember to FEED each placeholder when running the graph!

### ------exp3_SoftmaxRegression_mnist.py    
Time to try something more interesting!    
In this demo, I build a sofmax regression model for the MNIST dataset.    
--no hiden layer    
--cost function: cross entropy loss    
--momentum optimizer, learning rate 0.1, momentum 0.9    
--batch size 100    
--maximum iteration 1000    
--no regularization, no dropout

### ------exp4_CNN_mnist.py    
Of course the CNN model cannot be missed!    
In this demo I build a CNN model for MNIST dataset.    
--hidden: 2 convolution layers, each of which followed by one max-pooling layer    
--output: one fully connected layer (with dropout)    
--cost function: cross entropy loss    
--Adam optimizer, learning rate 1e-4    
--batch size 100    
--maximum iteration 20000    
--using dropout, dropout probability 0.5 (remember using dropout just in training process, no dropout when testing!)

(To be continued/////2017/3/19)
