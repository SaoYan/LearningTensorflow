#! /usr/bin/python

'''
import user's own dataset using high level API ---- From TFrecords

reference:
https://www.tensorflow.org/programmers_guide/datasets
'''

import tensorflow as tf
import numpy as np
import glob
import cv2
import os
import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _parse_function(example_proto):
  feature = {
        'image_raw': tf.FixedLenFeature(180*180*1, tf.float32),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64)
  }
  parsed_features = tf.parse_single_example(example_proto, feature)
  image = parsed_features['image_raw']
  image = tf.reshape(image, [180,180,1])
  return image

if __name__ == "__main__":
    dataset_name = 'dataset.tfrecords'

    ## save my own images to dataset (100 images of size 180 x 180)
    print("saving data ...\n")
    # build tfrecord file
    writer = tf.python_io.TFRecordWriter(dataset_name)
    # reader for original data
    h5f = h5py.File('my_data.h5', 'r')
    keys = list(h5f.keys())
    # save my images to dataset
    for key in keys:
        img = np.array(h5f[key]).astype(dtype=np.float32)
        height = img.shape[0]
        width = img.shape[1]
        feature = {
            'image_raw': _float_feature(img.reshape( (height*width) )),
            'height': _int64_feature(height),
            'width':  _int64_feature(width)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    h5f.close()
    writer.close()

    ## load the dataset and display them with tensorboard
    print("loading data ...\n")
    # 1. build Dataset object
    dataset = tf.data.TFRecordDataset(dataset_name)
    # 2. parsing TFrecords
    dataset = dataset.map(_parse_function)
    # 3. multiple epochs & batching
    dataset = dataset.repeat(10) # 10 epoches
    dataset = dataset.batch(64) # batch size: 64
    # 4. shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # construct iterator
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # read data
    img_batch = tf.placeholder(tf.float32, [None, 180, 180, 1])
    tf.summary.image(name='display', tensor=img_batch, max_outputs=6)
    step = 0
    with tf.Session() as sess:
        summary = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter('logs', sess.graph)
        while True:
            try:
                batch_img = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                print('\nEnd of dataset\n')
                break
            summ = sess.run(summary, {img_batch: batch_img})
            summ_writer.add_summary(summ, step)
            summ_writer.flush()
            step += 1
