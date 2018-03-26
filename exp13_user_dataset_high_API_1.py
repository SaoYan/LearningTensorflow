#! /usr/bin/python

'''
import user's own dataset using high level API ---- From Tensor

reference:
https://www.tensorflow.org/programmers_guide/datasets
'''

import tensorflow as tf
import numpy as np
import glob
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    ## load all the images into memory
    print('loading data into memory ...\n')
    data_path = 'my_data'
    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()
    Img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    Img = np.expand_dims(Img, 2) # 180 x 180 x 1
    Img = np.expand_dims(Img, 0) # 1 x 180 x 180 x 1
    Label = np.random.randint(0,2, size=(1,1))
    for i in range(1, len(files)):
        img = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 2) # 180 x 180 x 1
        img = np.expand_dims(img, 0) # 1 x 180 x 180 x 1
        label = np.random.randint(0,2, size=(1,1))
        Img = np.concatenate((Img,img), 0)
        Label = np.concatenate((Label,label), 0)

    ## construct dataset & iterator
    print('constructing Dataset & Iterator ...\n')
    dataset = tf.data.Dataset.from_tensor_slices((Img, Label))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    ## read data
    print('reading dataset ...\n')
    img_batch = tf.placeholder(tf.float32, [None, 180, 180, 1])
    tf.summary.image(name='display', tensor=img_batch, max_outputs=4)
    step = 0
    with tf.Session() as sess:
        summary = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter('logs', sess.graph)
        while True:
            try:
                batch = sess.rum(next_element)
            except tf.errors.OutOfRangeError:
                print('End of dataset\n')
                break
            summ = sess.run(summary, {img_batch: batch})
            summ_writer.add_summary(summ, step)
            summ_writer.flush()
            step += 1
