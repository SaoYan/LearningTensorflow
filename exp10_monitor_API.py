#! /usr/bin/python

'''
this code is a modification of Exp5
reference:
https://www.tensorflow.org/get_started/monitors
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# change the default logging level
# for details about logging level, see
#  https://www.tensorflow.org/get_started/monitors#enabling_logging_with_tensorflow
# tf.logging.set_verbosity(tf.logging.ERROR) # in case you don't want to see warnings(this will disable the display of logs!)
tf.logging.set_verbosity(tf.logging.INFO)  # in case you want to see log infor(this will also allow the warings to be shown!)

# Data sets
IRIS_TRAINING = "./IRIS_data/iris_training.csv"
IRIS_TEST = "./IRIS_data/iris_test.csv"

# Load datasets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# set monitor
validation_metrics = {
    "accuracy":
        tf.contrib.learn.metric_spec.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
            CLASSES),
    "precision":
        tf.contrib.learn.metric_spec.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
            CLASSES),
    "recall":
        tf.contrib.learn.metric_spec.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
            CLASSES)
}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

# Build 3 layer DNN with 10, 20, 10 units respectively
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="./IRIS_logs",
                                            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

# Fit model
# tf.contrib.learn offers a variety of predefined models, called Estimators
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])

# Evaluate accuracy
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
