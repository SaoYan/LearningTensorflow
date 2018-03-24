#! /usr/bin/python

'''
import user's own dataset using low level API
reference:
https://www.tensorflow.org/api_guides/python/reading_data
'''

import tensorflow as tf
import glob
import cv2
import os

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_tfrecord(tf_filename, size):
    queue = tf.train.string_input_producer([tf_filename])
    reader = tf.TFRecordReader()
    __, serialized_example = reader.read(queue)
    feature = {
          'image_raw': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, size)
    return image

if __name__ == "__main__":
    data_path = 'my_data'
    dataset_name = 'dataset.tfrecords'
    ###########################################
    # save my own images to dataset (400 images of size 180 x 180)
    ###########################################
    print("saving data ...\n")
    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()
    # build tfrecord file
    writer = tf.python_io.TFRecordWriter(dataset_name)
    # save my images to dataset
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE) # gray scale image
        image_raw = img.tostring() # save as 8bit data
        feature = {
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(img.shape[0]),
            'width':  _int64_feature(img.shape[1])
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

    ###########################################
    # load the dataset and display them with tensorboard
    ###########################################
    print("loading data ...\n")
    # define batch
    batch_size = 64
    Image = read_tfrecord(dataset_name, size=[180,180,1])
    # min_after_dequeue defines how big a buffer we will randomly sample from
    #    -- bigger means better shuffling but slower start up and more memory used.
    # capacity must be larger than min_after_dequeue
    #    -- recommendation:
    #   capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
    data_batch = tf.train.shuffle_batch([Image],
                batch_size = batch_size,
                capacity = 1000 + 3 * batch_size,
                num_threads = 2,
                min_after_dequeue = 1000)
    # placeholder
    img_batch = tf.placeholder(tf.uint8, [None, 180, 180, 1])
    # summary
    # we only display 6 images within each batch
    tf.summary.image(name='display', tensor=img_batch, max_outputs=6)
    # begin loading
    epoch = 0
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter('logs', sess.graph)
        while epoch < 10:
            print("epoch %d" % epoch)
            batch = sess.run(data_batch)
            summ = sess.run(summary, {img_batch: batch})
            summ_writer.add_summary(summ, epoch)
            summ_writer.flush()
            epoch += 1
        coord.request_stop()
        coord.join(threads)
