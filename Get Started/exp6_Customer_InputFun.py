#! /usr/bin/python

'''
in exp5, no manipulation of source data is required;
but in cases where more feature engineering is needed,
tf.contrib.learn supports using a custom input function;
in this function you can do your own preprocessing work

for more detail, see the following URL
https://www.tensorflow.org/get_started/input_fn

two important things worth paying attention on:
1) if our feature/label data is stored in pandas dataframes or numpy arrays, how to convert it to Tensors?
see https://www.tensorflow.org/get_started/input_fn#converting_feature_data_to_tensors
2) how to pass input_fn Data to Your Model?
see https://www.tensorflow.org/get_started/input_fn#passing_input_fn_data_to_your_model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

# set logging verbosity to INFO for more detailed log output
# see https://www.tensorflow.org/get_started/monitors#enabling_logging_with_tensorflow
tf.logging.set_verbosity(tf.logging.INFO)

# Importing the Housing Data
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("./BOSTON_data/boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("./BOSTON_data/boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("./BOSTON_data/boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)

# define FeatureColumns
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

# creating the DNNRegressor model
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          model_dir="./BOSTON_logs")

# build the input function
def my_input_fun(data_set):
    # preprocessing
    # for example: feature scaling
    # in this exp, none is needed

    # return
    # 1) a dict: (key)feature columns->(value)Tensors with the corresponding feature data
    # 2) a Tensor containing labels
    feature_cols = {k: tf.constant(data_set[k].values)
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels

# training
# see log output for INFO on training loss(every 100 steps)
regressor.fit(input_fn=lambda: my_input_fun(training_set), steps=5000)

# evaluating
ev = regressor.evaluate(input_fn=lambda: my_input_fun(test_set), steps=1)
loss_score = ev['loss']
print("Loss: %f"%loss_score)
# or
# print("Loss: {0:f}".format(loss_score))

# making prediction
y = regressor.predict(input_fn=lambda: my_input_fun(prediction_set))
# predict() returns an iterator; convert to a list and print predictions
# itertools.islice(y, 6): get the first 6 items from the iterator
predictions = list(itertools.islice(y, 6))
print ("Predictions: %s"%(str(predictions)))
